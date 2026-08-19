"""
autodock_benchmark.py

Benchmark Bio-JEPA scoring vs AutoDock Vina docking sur 50 molécules ChEMBL251
en utilisant la structure du récepteur A2A (PDB : 3EML).

Pipeline :
  1. Charge 50 molécules aléatoires depuis ChEMBL251
  2. Mesure le temps de scoring Bio-JEPA (encoding + sonde MLP)
  3. Tente un docking AutoDock Vina (mode simulation si binaire absent)
  4. Extrapole les temps à 1 000 et 1 000 000 de molécules
  5. Calcule le Pearson r de chaque méthode vs pChEMBL
  6. Affiche un tableau comparatif
  7. Sauvegarde results/autodock_benchmark.json

Dépendances optionnelles (pour docking réel) :
    pip install meeko gemmi vina

Usage :
    python autodock_benchmark.py
    python autodock_benchmark.py --checkpoint checkpoints/best_model.pt
    python autodock_benchmark.py --n-mols 50 --seed 0
    python autodock_benchmark.py --force-simulation
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader

from data.chembl_dataset import ChEMBLDataset
from data.mol_graph import smiles_vers_graphe
from evaluation.metrics import SondeMLP, extraire_embeddings
from models.bio_jepa import BioJEPA


# ── Constantes ────────────────────────────────────────────────────────────────

# Estimations littérature pour le mode simulation
VINA_CPU_SEC_PER_MOL      = 2.0     # s/mol — AutoDock Vina CPU 1 thread
AUTODOCK_GPU_SEC_PER_MOL  = 0.15    # s/mol — AutoDock-GPU (référence)
VINA_PEARSON_LITERATURE   = 0.35    # Pearson r typique Vina vs pChEMBL sur GPCRs

# Boîte de docking 3EML — centroïde du ligand ZMA
DOCKING_CENTER = (-11.8, -7.7, 60.5)   # Å  (calculé depuis les coords HETATM ZMA)
DOCKING_SIZE   = (22.0,  22.0, 22.0)   # Å

PDB_URL   = "https://files.rcsb.org/download/3EML.pdb"
PDB_CACHE = "data/3EML.pdb"

# Cache de la sonde MLP (indépendant de repositioning.py)
PROBE_CACHE = "checkpoints/probe_benchmark.pt"

# Dimension de sortie du GNNEncoder : 2 × embedding_dim = 2 × 256 = 512
ENCODER_OUT_DIM = 512


# ── Chargement du modèle ──────────────────────────────────────────────────────

def resoudre_checkpoint(path: str) -> str:
    if os.path.exists(path):
        return path
    for fallback in ["checkpoints/zinc_pretrained.pt", "checkpoints/best_model.pt"]:
        if os.path.exists(fallback):
            print(f"  Checkpoint introuvable → fallback : {fallback}")
            return fallback
    raise FileNotFoundError(
        f"Aucun checkpoint trouvé ({path}).\n"
        "Lancez d'abord :  python train.py --target CHEMBL251"
    )


def charger_modele(checkpoint_path: str, device: torch.device) -> BioJEPA:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = BioJEPA()
    if isinstance(ckpt, dict) and 'target_encoder' in ckpt:
        model.context_encoder.load_state_dict(ckpt['context_encoder'])
        model.target_encoder.load_state_dict(ckpt['target_encoder'])
        model.predictor.load_state_dict(ckpt['predictor'])
    else:
        model.load_state_dict(ckpt)
    return model.to(device).eval()


# ── Sonde MLP ─────────────────────────────────────────────────────────────────

def charger_ou_entrainer_sonde(
    model: BioJEPA,
    dataset,
    device: torch.device,
) -> SondeMLP:
    """Charge la sonde depuis PROBE_CACHE ou l'entraîne et la met en cache."""
    if os.path.exists(PROBE_CACHE):
        print(f"  Sonde chargée depuis {PROBE_CACHE}")
        sonde = SondeMLP(embedding_dim=ENCODER_OUT_DIM)
        sonde.load_state_dict(
            torch.load(PROBE_CACHE, map_location=device, weights_only=False)
        )
        return sonde.to(device).eval()

    print("  Entraînement de la sonde MLP sur l'ensemble du dataset...")
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    embeddings, labels = extraire_embeddings(model, loader, device, encoder='target')
    print(f"  Embeddings extraits : {embeddings.shape}")

    n = len(embeddings)
    rng = np.random.default_rng(0)
    idx = rng.permutation(n)
    split = int(0.8 * n)
    train_idx, val_idx = idx[:split], idx[split:]

    X_tr = torch.tensor(embeddings[train_idx], dtype=torch.float32).to(device)
    y_tr = torch.tensor(labels[train_idx],     dtype=torch.float32).to(device)
    X_va = torch.tensor(embeddings[val_idx],   dtype=torch.float32).to(device)
    y_va = torch.tensor(labels[val_idx],       dtype=torch.float32).to(device)

    sonde = SondeMLP(embedding_dim=embeddings.shape[1]).to(device)
    opt   = torch.optim.Adam(sonde.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=150)

    best_loss, best_state = float('inf'), None
    for epoch in range(150):
        sonde.train()
        opt.zero_grad()
        F.mse_loss(sonde(X_tr).squeeze(), y_tr).backward()
        opt.step()
        scheduler.step()

        sonde.eval()
        with torch.no_grad():
            val_loss = F.mse_loss(sonde(X_va).squeeze(), y_va).item()
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.clone() for k, v in sonde.state_dict().items()}

        if (epoch + 1) % 30 == 0:
            print(f"    epoch {epoch+1:>3}/150  val_loss={val_loss:.4f}")

    sonde.load_state_dict(best_state)
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(sonde.state_dict(), PROBE_CACHE)
    print(f"  Sonde sauvegardée → {PROBE_CACHE}")
    return sonde.to(device).eval()


# ── Timing Bio-JEPA ───────────────────────────────────────────────────────────

def mesurer_biojep(
    model: BioJEPA,
    sonde: SondeMLP,
    molecules: list,
    device: torch.device,
) -> tuple[np.ndarray, float]:
    """
    Encode 'molecules' (liste de PyG Data) et prédit pChEMBL.
    Retourne (predictions, elapsed_s).
    """
    batch = Batch.from_data_list(molecules).to(device)

    # Warmup : une passe avant la mesure
    with torch.no_grad():
        _ = model.encode(batch, encoder='target')
    _sync(device)

    t0 = time.perf_counter()
    with torch.no_grad():
        emb   = model.encode(batch, encoder='target')
        preds = sonde(emb).squeeze()
    _sync(device)
    elapsed = time.perf_counter() - t0

    return preds.cpu().numpy(), elapsed


def _sync(device: torch.device):
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elif device.type == 'mps':
        try:
            torch.mps.synchronize()
        except AttributeError:
            pass


# ── Préparation 3EML (récepteur) ──────────────────────────────────────────────

# Correspondance élément → type AutoDock4
_AD_TYPES = {
    'C': 'C', 'N': 'NA', 'O': 'OA', 'S': 'SA', 'H': 'H',
    'P': 'P', 'F': 'F', 'CL': 'Cl', 'BR': 'Br', 'I': 'I',
    'ZN': 'Zn', 'FE': 'Fe', 'MG': 'Mg', 'CA': 'Ca',
}


def telecharger_pdb(url: str, dest: str):
    if os.path.exists(dest):
        return
    os.makedirs(os.path.dirname(dest) or '.', exist_ok=True)
    print(f"  Téléchargement {url} → {dest}")
    urllib.request.urlretrieve(url, dest)


def pdb_vers_pdbqt_receptor(pdb_path: str, pdbqt_path: str):
    """
    Conversion PDB → PDBQT pour récepteur rigide (charges = 0, types génériques).
    Exclut ZMA (ligand) et HOH/WAT (eau).
    """
    with open(pdb_path) as f:
        lignes = f.readlines()

    out_lignes = []
    for ligne in lignes:
        record = ligne[:6].strip()
        if record not in ('ATOM', 'HETATM'):
            continue
        res_name = ligne[17:20].strip()
        if res_name in ('ZMA', 'HOH', 'WAT'):
            continue

        # Déduire l'élément (col 77-78 si disponible, sinon nom d'atome)
        elem = ligne[76:78].strip().upper() if len(ligne) > 76 else ''
        if not elem:
            atom_name = ligne[12:16].strip()
            elem = ''.join(c for c in atom_name if c.isalpha()).upper()[:2]
        ad_type = _AD_TYPES.get(elem, 'C')

        # Format PDBQT : cols 67-70 = espaces, 71-76 = charge (6 chars),
        # 77 = espace, 78-79 = type AutoDock (2 chars).
        # f"{0.0:6.3f}" = " 0.000" (6 chars, leading space inclus).
        pdbqt_ligne = ligne[:66].rstrip().ljust(66) + f"    {0.0:6.3f} {ad_type:<2}\n"
        out_lignes.append(pdbqt_ligne)

    with open(pdbqt_path, 'w') as f:
        f.writelines(out_lignes)
    print(f"  Récepteur PDBQT → {pdbqt_path}  ({len(out_lignes)} atomes)")


# ── Préparation ligand (meeko) ────────────────────────────────────────────────

def _assurer_meeko():
    """Installe meeko (+ gemmi) si absent de l'environnement courant."""
    try:
        import meeko  # noqa: F401
    except ImportError:
        print("  meeko non trouvé — installation via pip...")
        subprocess.check_call(
            [sys.executable, '-m', 'pip', 'install', '--quiet', 'meeko', 'gemmi'],
        )
        print("  meeko installé.")


def preparer_ligand_pdbqt(smiles: str) -> str:
    """
    SMILES → PDBQT via meeko.

    Pipeline :
      1. SMILES → RDKit mol  (fragment le plus lourd si sel/multi-composant)
      2. Ajout des hydrogènes (AddHs)
      3. Génération des coordonnées 3D (ETKDGv3 → fallback générique)
      4. Optimisation MMFF
      5. Préparation meeko  (MoleculePreparation.prepare)
      6. Sérialisation PDBQT (write_pdbqt_string)

    Retourne la chaîne PDBQT.
    Lève RuntimeError avec message explicite si une étape échoue
    (pour être attrapée par l'appelant avec le compteur d'échecs).
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem
    import warnings

    # ── Étape 1 : SMILES → mol (fragment le plus lourd) ──────────────────────
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise RuntimeError(f"MolFromSmiles a retourné None")

    frags = Chem.GetMolFrags(mol, asMols=True)
    if len(frags) > 1:
        mol = max(frags, key=lambda m: m.GetNumAtoms())

    # ── Étape 2 : ajout des hydrogènes ───────────────────────────────────────
    mol = Chem.AddHs(mol)

    # ── Étape 3 : coordonnées 3D ─────────────────────────────────────────────
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    result = AllChem.EmbedMolecule(mol, params)
    if result == -1:
        result = AllChem.EmbedMolecule(mol, randomSeed=42)
    if result == -1:
        raise RuntimeError("EmbedMolecule échoué (ETKDGv3 + fallback générique)")

    # ── Étape 4 : optimisation MMFF ──────────────────────────────────────────
    AllChem.MMFFOptimizeMolecule(mol)   # -1 = non applicable, ignoré

    # ── Étapes 5-6 : meeko ────────────────────────────────────────────────────
    from meeko import MoleculePreparation

    preparator = MoleculePreparation()
    preparator.prepare(mol)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', DeprecationWarning)
        pdbqt_str = preparator.write_pdbqt_string()

    if not pdbqt_str:
        raise RuntimeError("write_pdbqt_string a retourné une chaîne vide")
    return pdbqt_str


# ── Docking Vina réel ─────────────────────────────────────────────────────────

def detecter_vina() -> Optional[str]:
    """Cherche le binaire AutoDock Vina dans le PATH."""
    import shutil
    for name in ('vina', 'autodock_vina', 'vina_1.2.5_linux_x86_64',
                 'vina_1.2.3_linux_x86_64'):
        path = shutil.which(name)
        if path:
            return path
    return None


def _parser_score_vina(stdout: str) -> Optional[float]:
    """Extrait le meilleur score (mode 1) depuis la sortie standard de Vina."""
    for line in stdout.splitlines():
        parts = line.strip().split()
        # Ligne de résultat : "   1      -8.3    0.000    0.000"
        if len(parts) >= 2 and parts[0] == '1':
            try:
                return float(parts[1])
            except ValueError:
                pass
    return None


def docking_une_molecule(
    receptor_pdbqt: str,
    ligand_pdbqt: str,
    vina_bin: str,
) -> Optional[float]:
    """Lance Vina sur un ligand et retourne le meilleur score (kcal/mol)."""
    cx, cy, cz = DOCKING_CENTER
    sx, sy, sz = DOCKING_SIZE

    with tempfile.NamedTemporaryFile(suffix='.pdbqt', mode='w', delete=False) as ftmp:
        ftmp.write(ligand_pdbqt)
        lig_path = ftmp.name

    out_path = lig_path.replace('.pdbqt', '_out.pdbqt')
    try:
        cmd = [
            vina_bin,
            '--receptor', receptor_pdbqt,
            '--ligand',   lig_path,
            '--center_x', str(cx),
            '--center_y', str(cy),
            '--center_z', str(cz),
            '--size_x',   str(sx),
            '--size_y',   str(sy),
            '--size_z',   str(sz),
            '--num_modes',      '1',
            '--exhaustiveness', '4',
            '--out', out_path,
        ]
        print(f"  [CMD] {' '.join(cmd)}")
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            print(f"  [STDOUT] {result.stdout}")
            print(f"  [STDERR] {result.stderr}")
        return _parser_score_vina(result.stdout)
    except Exception:
        return None
    finally:
        for p in (lig_path, out_path):
            try:
                os.unlink(p)
            except OSError:
                pass


def mesurer_vina_reel(
    smiles_list: list,
    pchembl_arr: np.ndarray,
    receptor_pdbqt: str,
    vina_bin: str,
) -> dict:
    """Docking réel Vina molécule par molécule. Retourne les stats."""
    scores, valides_pchembl = [], []
    elapsed_total = 0.0
    n = len(smiles_list)

    _assurer_meeko()
    print(f"\n  [Vina réel] {n} molécules  (exhaustiveness=4)...")
    echecs = 0
    for i, (smi, pc) in enumerate(zip(smiles_list, pchembl_arr)):
        try:
            pdbqt = preparer_ligand_pdbqt(smi)
        except Exception as e:
            if echecs < 3:
                import traceback
                print(f"  [DEBUG mol {i+1}] {traceback.format_exc()}")
            echecs += 1
            continue

        t0 = time.perf_counter()
        score = docking_une_molecule(receptor_pdbqt, pdbqt, vina_bin)
        elapsed_total += time.perf_counter() - t0

        if score is not None:
            scores.append(score)
            valides_pchembl.append(pc)
            print(f"    mol {i+1:>3}/{n}  score={score:>6.2f} kcal/mol"
                  f"  pChEMBL={pc:.2f}", end='\r')

    if echecs:
        print(f"\n  [prep] {echecs}/{n} molécules ont échoué à la préparation PDBQT")

    print()
    scores_arr       = np.array(scores)
    pchembl_valides  = np.array(valides_pchembl)
    n_valide         = len(scores_arr)

    # Vina : score négatif → forte affinité ; on inverse avant corrélation
    r_val = float('nan')
    if n_valide >= 4:
        r_val, _ = pearsonr(-scores_arr, pchembl_valides)

    return {
        'mode'      : 'reel',
        'n_valid'   : n_valide,
        'elapsed_s' : elapsed_total,
        's_per_mol' : elapsed_total / max(n_valide, 1),
        'pearson_r' : r_val,
        'scores'    : scores_arr.tolist(),
    }


# ── Mode simulation ────────────────────────────────────────────────────────────

def mesurer_vina_simulation(n_mols: int) -> dict:
    """
    Estime le temps et le Pearson r de Vina depuis la littérature.
    Utilisé quand le binaire AutoDock Vina n'est pas disponible.
    """
    elapsed_s = VINA_CPU_SEC_PER_MOL * n_mols
    return {
        'mode'      : 'simulation',
        'n_valid'   : n_mols,
        'elapsed_s' : elapsed_s,
        's_per_mol' : VINA_CPU_SEC_PER_MOL,
        'pearson_r' : VINA_PEARSON_LITERATURE,
        'note'      : (
            f"SIMULATION — AutoDock Vina non disponible sur ce système "
            f"(pip install vina échoue sur macOS ARM). "
            f"Timing estimé : ~{VINA_CPU_SEC_PER_MOL} s/mol (CPU, 1 thread, littérature). "
            f"Pearson r : ~{VINA_PEARSON_LITERATURE:.2f} "
            f"(valeur typique GPCR/kinase, littérature)."
        ),
    }


# ── Extrapolation ─────────────────────────────────────────────────────────────

def extrapoler(elapsed_s: float, n_base: int) -> dict:
    """Extrapolation linéaire : temps pour 1 000 et 1 000 000 molécules."""
    s_per = elapsed_s / max(n_base, 1)
    return {
        'n_base'   : n_base,
        '1000'     : s_per * 1_000,
        '1000000'  : s_per * 1_000_000,
    }


# ── Affichage ─────────────────────────────────────────────────────────────────

def _fmt(s: float) -> str:
    """Formate une durée en secondes de manière lisible."""
    if s < 60:
        return f"{s:.1f} s"
    if s < 3600:
        return f"{s/60:.1f} min"
    if s < 86400:
        return f"{s/3600:.1f} h"
    return f"{s/86400:.1f} j"


def afficher_tableau(n_mols: int, bj: dict, vina: dict):
    ex_bj   = extrapoler(bj['elapsed_s'],   n_mols)
    ex_vina = extrapoler(vina['elapsed_s'], n_mols)

    W = 76
    sep  = "─" * W
    sep2 = "═" * W

    print(f"\n{sep2}")
    print("  TABLEAU COMPARATIF : Bio-JEPA vs AutoDock Vina")
    if vina['mode'] == 'simulation':
        print("  ⚠  Vina : MODE SIMULATION (binaire non disponible — estimation littérature)")
    print(sep2)
    print(f"  {'Méthode':<26} {'50 mols':>8} {'1 000 mols':>12} "
          f"{'1 M mols':>12} {'Pearson r':>10}")
    print(sep)

    def _ligne(label, ex, r):
        return (
            f"  {label:<26}"
            f" {_fmt(ex_bj['n_base'] * ex['n_base'] / n_mols if False else bj['elapsed_s'] if label.startswith('Bio') else vina['elapsed_s']):>8}"
            f" {_fmt(ex['1000']):>12}"
            f" {_fmt(ex['1000000']):>12}"
            f" {r:>10.3f}"
        )

    print(
        f"  {'Bio-JEPA  (scoring)':<26}"
        f" {_fmt(bj['elapsed_s']):>8}"
        f" {_fmt(ex_bj['1000']):>12}"
        f" {_fmt(ex_bj['1000000']):>12}"
        f" {bj['pearson_r']:>10.3f}"
    )

    vina_label = (
        "AutoDock Vina (réel)"
        if vina['mode'] == 'reel'
        else "AutoDock Vina (sim)"
    )
    print(
        f"  {vina_label:<26}"
        f" {_fmt(vina['elapsed_s']):>8}"
        f" {_fmt(ex_vina['1000']):>12}"
        f" {_fmt(ex_vina['1000000']):>12}"
        f" {vina['pearson_r']:>10.3f}"
    )
    print(sep)

    # Accélération
    accel = (vina['elapsed_s'] / bj['elapsed_s']) if bj['elapsed_s'] > 0 else float('inf')
    delta_r = bj['pearson_r'] - vina['pearson_r']
    print(f"\n  Accélération Bio-JEPA vs Vina : ×{accel:,.0f}  "
          f"(ΔPearson r = {delta_r:+.3f})")
    print(f"  n = {n_mols} molécules  ·  récepteur = 3EML (A2A adenosine)")
    print(sep2)


# ── Sauvegarde JSON ────────────────────────────────────────────────────────────

def sauvegarder(path: str, data: dict):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"\n  Résultats sauvegardés → {path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Benchmark Bio-JEPA vs AutoDock Vina (A2A, 3EML)',
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        '--checkpoint', default='checkpoints/zinc_pretrained.pt',
        help='Checkpoint Bio-JEPA (défaut: checkpoints/zinc_pretrained.pt)',
    )
    parser.add_argument('--target',   default='CHEMBL251',
                        help='Target ChEMBL pour le dataset (défaut: CHEMBL251)')
    parser.add_argument('--csv',      default=None,
                        help='CSV local (si pas de connexion API)')
    parser.add_argument('--n-mols',   type=int, default=50,
                        help='Nombre de molécules à benchmarker (défaut: 50)')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--seed',     type=int, default=42,
                        help='Graine pour la sélection aléatoire')
    parser.add_argument('--force-simulation', action='store_true',
                        help='Forcer le mode simulation Vina (ignorer le binaire)')
    parser.add_argument('--out', default='results/autodock_benchmark.json',
                        help='Chemin de sortie JSON')
    args = parser.parse_args()

    # ── Dispositif ──────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"[Dispositif] {device}")

    # ── Modèle ──────────────────────────────────────────────────────────────
    print(f"\n[Modèle]")
    ckpt_path = resoudre_checkpoint(args.checkpoint)
    model = charger_modele(ckpt_path, device)
    n_params = sum(p.numel() for p in model.target_encoder.parameters())
    print(f"  {ckpt_path}  ({n_params:,} paramètres — target encoder)")

    # ── Dataset ──────────────────────────────────────────────────────────────
    print(f"\n[Données] {args.target}")
    dataset = ChEMBLDataset(
        root='data/chembl',
        csv_path=args.csv,
        target_chembl_id=args.target,
    )
    print(f"  {len(dataset):,} molécules")

    # ── Sonde MLP ────────────────────────────────────────────────────────────
    print(f"\n[Sonde MLP]")
    sonde = charger_ou_entrainer_sonde(model, dataset, device)

    # ── Sélection des molécules ───────────────────────────────────────────────
    rng   = np.random.default_rng(args.seed)
    n     = min(args.n_mols, len(dataset))
    idx   = rng.choice(len(dataset), size=n, replace=False)

    molecules    = [dataset[int(i)] for i in idx]
    smiles_list  = [dataset[int(i)].smiles for i in idx]
    pchembl_arr  = np.array([float(dataset[int(i)].y) for i in idx])

    print(f"\n[Sélection] {n} molécules aléatoires (seed={args.seed})")
    print(f"  pChEMBL : min={pchembl_arr.min():.2f}  max={pchembl_arr.max():.2f}"
          f"  mean={pchembl_arr.mean():.2f}  std={pchembl_arr.std():.2f}")

    # ── Benchmark Bio-JEPA ────────────────────────────────────────────────────
    print(f"\n[Bio-JEPA] Scoring de {n} molécules...")
    preds_bj, elapsed_bj = mesurer_biojep(model, sonde, molecules, device)
    r_bj, _ = pearsonr(preds_bj, pchembl_arr)
    ms_mol = elapsed_bj / n * 1000
    print(f"  Temps total : {elapsed_bj*1000:.1f} ms  ({ms_mol:.3f} ms/mol)")
    print(f"  Pearson r   : {r_bj:.3f}")

    bj_result = {
        'elapsed_s'   : elapsed_bj,
        's_per_mol'   : elapsed_bj / n,
        'pearson_r'   : float(r_bj),
        'predictions' : preds_bj.tolist(),
        'pchembl_gt'  : pchembl_arr.tolist(),
    }

    # ── Benchmark AutoDock Vina ───────────────────────────────────────────────
    print(f"\n[AutoDock Vina]")
    vina_bin = None if args.force_simulation else detecter_vina()

    if vina_bin:
        print(f"  Binaire Vina détecté : {vina_bin}")
        telecharger_pdb(PDB_URL, PDB_CACHE)
        receptor_pdbqt = PDB_CACHE.replace('.pdb', '.pdbqt')
        pdb_vers_pdbqt_receptor(PDB_CACHE, receptor_pdbqt)
        print(f"  Récepteur PDBQT : {os.path.getsize(receptor_pdbqt):,} bytes")
        vina_result = mesurer_vina_reel(smiles_list, pchembl_arr, receptor_pdbqt, vina_bin)
        print(f"  Temps total : {_fmt(vina_result['elapsed_s'])}  "
              f"({vina_result['s_per_mol']:.2f} s/mol)")
        print(f"  Pearson r   : {vina_result['pearson_r']:.3f}  "
              f"(n valide = {vina_result['n_valid']})")
    else:
        if args.force_simulation:
            print("  Mode simulation forcé.")
        else:
            print("  Binaire vina introuvable (pip install vina échoue sur macOS ARM).")
            print("  → Mode simulation (estimations littérature).")
        vina_result = mesurer_vina_simulation(n)
        print(f"  Temps estimé : {_fmt(vina_result['elapsed_s'])}  "
              f"({vina_result['s_per_mol']:.1f} s/mol)")
        print(f"  Pearson r    : {vina_result['pearson_r']:.3f}  (littérature)")

    # ── Tableau ───────────────────────────────────────────────────────────────
    afficher_tableau(n, bj_result, vina_result)

    # ── JSON ──────────────────────────────────────────────────────────────────
    ex_bj   = extrapoler(elapsed_bj,                 n)
    ex_vina = extrapoler(vina_result['elapsed_s'],   n)

    accel = (vina_result['elapsed_s'] / elapsed_bj) if elapsed_bj > 0 else None

    output = {
        'config': {
            'checkpoint'      : ckpt_path,
            'target'          : args.target,
            'n_mols'          : n,
            'seed'            : args.seed,
            'device'          : str(device),
            'receptor_pdb'    : '3EML',
            'docking_center'  : list(DOCKING_CENTER),
            'docking_size'    : list(DOCKING_SIZE),
        },
        'bio_jepa': {
            'elapsed_s'        : bj_result['elapsed_s'],
            'ms_per_mol'       : bj_result['s_per_mol'] * 1000,
            'pearson_r'        : bj_result['pearson_r'],
            'extrapolation': {
                '50_mols_s'       : bj_result['elapsed_s'],
                '1000_mols_s'     : ex_bj['1000'],
                '1000000_mols_s'  : ex_bj['1000000'],
                '1000_mols_fmt'   : _fmt(ex_bj['1000']),
                '1000000_mols_fmt': _fmt(ex_bj['1000000']),
            },
            'predictions' : bj_result['predictions'],
        },
        'autodock_vina': {
            'mode'             : vina_result['mode'],
            'elapsed_s'        : vina_result['elapsed_s'],
            's_per_mol'        : vina_result['s_per_mol'],
            'n_valid'          : vina_result['n_valid'],
            'pearson_r'        : vina_result['pearson_r'],
            'extrapolation': {
                '50_mols_s'       : vina_result['elapsed_s'],
                '1000_mols_s'     : ex_vina['1000'],
                '1000000_mols_s'  : ex_vina['1000000'],
                '1000_mols_fmt'   : _fmt(ex_vina['1000']),
                '1000000_mols_fmt': _fmt(ex_vina['1000000']),
            },
            **({'note': vina_result['note']} if 'note' in vina_result else {}),
            **({'scores': vina_result['scores']} if 'scores' in vina_result else {}),
        },
        'comparison': {
            'speedup_biojep_vs_vina' : accel,
            'delta_pearson_r'        : float(r_bj) - vina_result['pearson_r'],
            'pchembl_ground_truth'   : pchembl_arr.tolist(),
            'smiles'                 : smiles_list,
        },
    }

    sauvegarder(args.out, output)
    print("\nFait.")


if __name__ == '__main__':
    main()
