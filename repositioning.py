"""
repositioning.py

Drug repositioning : prédit l'affinité sur CHEMBL251 (récepteur A2A) pour
les médicaments FDA approuvés encodés par Bio-JEPA.

Pipeline :
  1. Télécharge les médicaments FDA (max_phase=4) :
       - Essai URL GitHub (approved_drugs.csv)
       - Fallback : API ChEMBL  →  molecule.filter(max_phase=4)
  2. Encode chaque SMILES avec le Target Encoder (zinc_pretrained.pt)
  3. Charge ou (ré)entraîne une sonde MLP sur ChEMBL251
       → sauvegardée dans checkpoints/probe_chembl251.pt pour les runs suivants
  4. Prédit le score pChEMBL pour chaque molécule FDA
  5. Sauvegarde le top-20 dans results/repositioning_results.json

Note : la sonde est toujours entraînée avec le MÊME encodeur que celui utilisé
pour les médicaments FDA — cohérence garantie même si l'encodeur change.

Usage :
    python repositioning.py
    python repositioning.py --checkpoint checkpoints/best_model.pt
    python repositioning.py --retrain-probe     # force le ré-entraînement de la sonde
    python repositioning.py --max-drugs 500     # limite le nombre de médicaments FDA
"""

import argparse
import json
import os
from datetime import date
from io import StringIO

import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader

from data.chembl_dataset import ChEMBLDataset, creer_dataloaders
from data.mol_graph import smiles_vers_graphe
from evaluation.metrics import SondeMLP, extraire_embeddings
from models.bio_jepa import BioJEPA


# ── Constantes ────────────────────────────────────────────────────────────────

FDA_CSV_URL = (
    "https://raw.githubusercontent.com/chembl/chembl_webresource_client"
    "/master/chembl_webresource_client/test_data/approved_drugs.csv"
)

# Mapping code ATC (1er caractère) → classe thérapeutique
ATC_CLASSES = {
    'A': 'Voies digestives / métabolisme',
    'B': 'Sang / organes hématopoïétiques',
    'C': 'Système cardiovasculaire',
    'D': 'Dermatologie',
    'G': 'Système génito-urinaire',
    'H': 'Préparations hormonales systémiques',
    'J': 'Anti-infectieux systémiques',
    'L': 'Antinéoplasiques / immunomodulateurs',
    'M': 'Système musculo-squelettique',
    'N': 'Système nerveux',
    'P': 'Antiparasitaires',
    'R': 'Système respiratoire',
    'S': 'Organes sensoriels',
    'V': 'Divers',
}


# ── Acquisition des médicaments FDA ──────────────────────────────────────────

def _telecharger_csv_github() -> pd.DataFrame | None:
    """Essaie de télécharger approved_drugs.csv depuis GitHub."""
    try:
        r = requests.get(FDA_CSV_URL, timeout=10)
        r.raise_for_status()
        df = pd.read_csv(StringIO(r.text))
        print(f"  GitHub : {len(df)} médicaments téléchargés")
        return df
    except Exception as e:
        print(f"  GitHub inaccessible ({e})")
        return None


def _telecharger_chembl_api(max_drugs: int | None) -> pd.DataFrame:
    """Télécharge les médicaments FDA via l'API ChEMBL (max_phase=4)."""
    from chembl_webresource_client.new_client import new_client
    print("  API ChEMBL (max_phase=4)...")

    mol_client = new_client.molecule
    queryset = mol_client.filter(max_phase=4).only([
        'molecule_chembl_id', 'pref_name',
        'molecule_structures', 'atc_classifications',
    ])

    records = []
    for m in queryset:
        struct = m.get('molecule_structures') or {}
        smi = struct.get('canonical_smiles', '')
        if not smi:
            continue

        atc = m.get('atc_classifications') or []
        indication = _atc_vers_indication(atc)

        records.append({
            'chembl_id'  : m.get('molecule_chembl_id', ''),
            'name'       : m.get('pref_name') or m.get('molecule_chembl_id', ''),
            'smiles'     : smi,
            'indication' : indication,
        })

        if max_drugs and len(records) >= max_drugs:
            break

    print(f"  {len(records)} médicaments valides récupérés")
    return pd.DataFrame(records)


def _atc_vers_indication(atc_list: list) -> str:
    """Convertit une liste de codes ATC en libellé thérapeutique."""
    if not atc_list:
        return 'Non spécifiée'
    classes = []
    for code in atc_list:
        c = str(code)[0].upper()
        if c in ATC_CLASSES and ATC_CLASSES[c] not in classes:
            classes.append(ATC_CLASSES[c])
    return ' / '.join(classes) if classes else 'Non spécifiée'


def _normaliser_df(df: pd.DataFrame) -> pd.DataFrame:
    """Harmonise les noms de colonnes entre les différentes sources."""
    rename = {}
    for col in df.columns:
        lc = col.lower()
        if lc in ('canonical_smiles', 'smiles', 'molecule_smiles'):
            rename[col] = 'smiles'
        elif lc in ('pref_name', 'molecule_name', 'drug_name', 'name'):
            rename[col] = 'name'
        elif lc in ('molecule_chembl_id', 'chembl_id'):
            rename[col] = 'chembl_id'
    df = df.rename(columns=rename)
    if 'indication' not in df.columns:
        df['indication'] = 'Non spécifiée'
    if 'chembl_id' not in df.columns:
        df['chembl_id'] = ''
    return df


def charger_medicaments_fda(max_drugs: int | None = None) -> pd.DataFrame:
    """
    Charge les médicaments FDA approuvés.

    Stratégie :
      1. URL GitHub (approved_drugs.csv)
      2. API ChEMBL  →  molecule.filter(max_phase=4)
    """
    print("[Médicaments FDA]")
    df = _telecharger_csv_github()

    if df is None or df.empty:
        df = _telecharger_chembl_api(max_drugs)
    else:
        df = _normaliser_df(df)

    # Nettoyage de base
    df = df.dropna(subset=['smiles'])
    df['smiles'] = df['smiles'].astype(str).str.strip()
    df = df[df['smiles'].str.len() > 0].reset_index(drop=True)

    if max_drugs:
        df = df.iloc[:max_drugs].reset_index(drop=True)

    print(f"  → {len(df)} médicaments retenus")
    return df


# ── Chargement du modèle ──────────────────────────────────────────────────────

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


# ── Encodage d'une liste de SMILES ───────────────────────────────────────────

def encoder_smiles(
    smiles_list: list,
    model: BioJEPA,
    device: torch.device,
    batch_size: int = 256,
) -> tuple[np.ndarray, list[int]]:
    """
    Encode une liste de SMILES avec le Target Encoder.

    Returns:
        embeddings      : [N_valides, embedding_dim]
        indices_valides : indices dans smiles_list dont le SMILES est parsable
    """
    graphes, indices_valides = [], []
    for i, smi in enumerate(smiles_list):
        g = smiles_vers_graphe(smi)
        if g is not None:
            graphes.append(g)
            indices_valides.append(i)

    if not graphes:
        return np.empty((0, model.embedding_dim)), []

    model.eval()
    embeddings_chunks = []
    with torch.no_grad():
        for start in range(0, len(graphes), batch_size):
            batch = Batch.from_data_list(graphes[start:start + batch_size]).to(device)
            z = model.encode(batch, encoder='target')
            embeddings_chunks.append(z.cpu().numpy())

    return np.vstack(embeddings_chunks), indices_valides


# ── Sonde MLP ────────────────────────────────────────────────────────────────

def entrainer_sonde(
    model: BioJEPA,
    chembl_loader: DataLoader,
    device: torch.device,
    embedding_dim: int = 256,
    hidden_dim: int = 256,
    lr: float = 1e-3,
    num_epochs: int = 200,
) -> SondeMLP:
    """
    Entraîne une sonde MLP sur les embeddings ChEMBL251 du modèle courant.

    Utilise systématiquement le MÊME encodeur que celui de l'inférence
    sur les médicaments FDA → cohérence train/test garantie.
    """
    print("  Extraction des embeddings ChEMBL251...")
    z, y = extraire_embeddings(model, chembl_loader, device, encoder='target')
    print(f"  {z.shape[0]} molécules  |  embedding_dim={z.shape[1]}")

    z_t = torch.tensor(z, dtype=torch.float, device=device)
    y_t = torch.tensor(y, dtype=torch.float, device=device)

    sonde = SondeMLP(embedding_dim, hidden_dim).to(device)
    opt = torch.optim.Adam(sonde.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=num_epochs)

    best_loss, best_state = float('inf'), None

    for epoch in range(num_epochs):
        sonde.train()
        opt.zero_grad()
        F.mse_loss(sonde(z_t), y_t).backward()
        opt.step()
        scheduler.step()

        with torch.no_grad():
            loss = F.mse_loss(sonde(z_t), y_t).item()
        if loss < best_loss:
            best_loss = loss
            best_state = {k: v.clone() for k, v in sonde.state_dict().items()}

        if (epoch + 1) % 50 == 0:
            print(f"    Epoch {epoch+1:>3}/{num_epochs}  loss={loss:.4f}")

    sonde.load_state_dict(best_state)
    sonde.eval()
    print(f"  Meilleure loss : {best_loss:.4f}")
    return sonde


def charger_ou_entrainer_sonde(
    probe_path: str,
    model: BioJEPA,
    chembl_loader: DataLoader,
    device: torch.device,
    force_retrain: bool = False,
    embedding_dim: int = 256,
) -> SondeMLP:
    """Charge la sonde depuis le cache ou (ré)entraîne si nécessaire."""
    if os.path.exists(probe_path) and not force_retrain:
        print(f"  Sonde chargée depuis {probe_path}")
        sonde = SondeMLP(embedding_dim).to(device)
        sonde.load_state_dict(
            torch.load(probe_path, map_location=device, weights_only=True)
        )
        sonde.eval()
        return sonde

    reason = "forcé" if force_retrain else "non trouvé"
    print(f"  Entraînement de la sonde ({reason})...")
    sonde = entrainer_sonde(model, chembl_loader, device, embedding_dim=embedding_dim)

    os.makedirs(os.path.dirname(probe_path) or '.', exist_ok=True)
    torch.save(sonde.state_dict(), probe_path)
    print(f"  Sonde sauvegardée → {probe_path}")
    return sonde


# ── Prédiction et résultats ───────────────────────────────────────────────────

def predire_affinites(
    sonde: SondeMLP,
    embeddings: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    sonde.eval()
    with torch.no_grad():
        z = torch.tensor(embeddings, dtype=torch.float, device=device)
        preds = sonde(z).cpu().numpy()
    return preds


def construire_resultats(
    df_fda: pd.DataFrame,
    indices_valides: list[int],
    scores: np.ndarray,
    top_k: int = 20,
) -> list[dict]:
    """Associe les scores aux médicaments et retourne le top-K trié."""
    rows = []
    for local_i, global_i in enumerate(indices_valides):
        row = df_fda.iloc[global_i]
        rows.append({
            'name'            : str(row.get('name', row.get('chembl_id', f'mol_{global_i}'))),
            'chembl_id'       : str(row.get('chembl_id', '')),
            'smiles'          : str(row['smiles']),
            'predicted_pchembl': round(float(scores[local_i]), 4),
            'indication'      : str(row.get('indication', 'Non spécifiée')),
        })

    rows.sort(key=lambda x: x['predicted_pchembl'], reverse=True)
    for rank, r in enumerate(rows[:top_k], start=1):
        r['rank'] = rank
    return rows[:top_k]


def afficher_top20(resultats: list[dict]):
    largeur = 78
    print("\n" + "═" * largeur)
    print("  TOP 20 — Repositionnement sur CHEMBL251 (récepteur A2A)")
    print("═" * largeur)
    print(f"  {'Rg':>3}  {'Nom':<25}  {'pChEMBL':>8}  Indication")
    print("  " + "─" * (largeur - 2))
    for r in resultats:
        nom = r['name'][:24] if len(r['name']) > 24 else r['name']
        indic = r['indication'][:38] if len(r['indication']) > 38 else r['indication']
        print(f"  {r['rank']:>3}  {nom:<25}  {r['predicted_pchembl']:>8.3f}  {indic}")
    print("═" * largeur)


# ── Point d'entrée ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Drug repositioning avec Bio-JEPA sur CHEMBL251',
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        '--checkpoint', default='checkpoints/zinc_pretrained.pt',
        help='Checkpoint encodeur Bio-JEPA (défaut: zinc_pretrained.pt)',
    )
    parser.add_argument(
        '--probe-checkpoint', default='checkpoints/probe_chembl251.pt',
        help='Cache de la sonde MLP (défaut: checkpoints/probe_chembl251.pt)',
    )
    parser.add_argument(
        '--target', default='CHEMBL251',
        help='Cible ChEMBL pour entraîner la sonde (défaut: CHEMBL251)',
    )
    parser.add_argument(
        '--retrain-probe', action='store_true',
        help='Force le ré-entraînement de la sonde même si le cache existe',
    )
    parser.add_argument(
        '--max-drugs', type=int, default=None,
        help='Limite le nombre de médicaments FDA traités (pour les tests rapides)',
    )
    parser.add_argument(
        '--top-k', type=int, default=20,
        help='Nombre de molécules dans le rapport final (défaut: 20)',
    )
    parser.add_argument(
        '--out-dir', default='results',
        help='Dossier de sortie (défaut: results/)',
    )
    args = parser.parse_args()

    # --- Dispositif ---
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"[Dispositif] {device}")

    os.makedirs(args.out_dir, exist_ok=True)

    # --- Encodeur ---
    print(f"\n[Encodeur] {args.checkpoint}")
    if not os.path.exists(args.checkpoint):
        fallback = 'checkpoints/best_model.pt'
        if os.path.exists(fallback):
            print(f"  zinc_pretrained.pt absent → fallback sur {fallback}")
            args.checkpoint = fallback
        else:
            raise FileNotFoundError(
                f"Aucun checkpoint trouvé.\n"
                "Lancez d'abord :  python pretrain.py  ou  python train.py --target CHEMBL251"
            )
    model = charger_modele(args.checkpoint, device)
    print(f"  Target Encoder : {sum(p.numel() for p in model.target_encoder.parameters()):,} params")

    # --- Sonde MLP ---
    print(f"\n[Sonde MLP] Cible : {args.target}")
    dataset = ChEMBLDataset(root='data/chembl', target_chembl_id=args.target)
    _, _, _ = creer_dataloaders(dataset, batch_size=256)  # split canonique
    full_loader = DataLoader(dataset, batch_size=256, shuffle=False)

    sonde = charger_ou_entrainer_sonde(
        probe_path=args.probe_checkpoint,
        model=model,
        chembl_loader=full_loader,
        device=device,
        force_retrain=args.retrain_probe,
        embedding_dim=model.embedding_dim,
    )

    # --- Médicaments FDA ---
    df_fda = charger_medicaments_fda(args.max_drugs)

    # --- Encodage ---
    print(f"\n[Encodage] {len(df_fda)} médicaments → Target Encoder...")
    embeddings, indices_valides = encoder_smiles(
        smiles_list=df_fda['smiles'].tolist(),
        model=model,
        device=device,
    )
    n_invalides = len(df_fda) - len(indices_valides)
    print(f"  {len(indices_valides)} SMILES valides  |  {n_invalides} ignorés")
    print(f"  Shape embeddings : {embeddings.shape}")

    # --- Prédiction ---
    print("\n[Prédiction] Scores pChEMBL sur CHEMBL251...")
    scores = predire_affinites(sonde, embeddings, device)
    print(f"  min={scores.min():.3f}  max={scores.max():.3f}"
          f"  mean={scores.mean():.3f}  std={scores.std():.3f}")

    # --- Top-K ---
    resultats = construire_resultats(df_fda, indices_valides, scores, top_k=args.top_k)
    afficher_top20(resultats)

    # --- Sauvegarde JSON ---
    out_path = os.path.join(args.out_dir, 'repositioning_results.json')
    output = {
        'metadata': {
            'encoder_checkpoint'   : args.checkpoint,
            'chembl_target'        : args.target,
            'n_fda_drugs_screened' : len(df_fda),
            'n_valid_smiles'       : len(indices_valides),
            'score_stats': {
                'min'  : round(float(scores.min()), 4),
                'max'  : round(float(scores.max()), 4),
                'mean' : round(float(scores.mean()), 4),
                'std'  : round(float(scores.std()), 4),
            },
            'date': str(date.today()),
        },
        'top20': resultats,
    }

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n[Résultats] Sauvegardés → {out_path}")
    print(f"  Top 1 : {resultats[0]['name']}  (pChEMBL prédit = {resultats[0]['predicted_pchembl']:.3f})")


if __name__ == '__main__':
    main()
