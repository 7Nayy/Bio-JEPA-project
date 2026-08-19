"""
ablation_study.py

Étude d'ablation de Bio-JEPA sur ChEMBL251 (récepteur A2A).

Teste 4 variantes du mécanisme JEPA :
  V1 — Bio-JEPA complet (référence)
  V2 — Sans EMA  : copie directe ctx → target à chaque batch (τ=0)
  V3 — Sans masquage progressif : masquage fixe à 30%
  V4 — Sans stop-gradient : gradients traversent le Target Encoder,
        target encoder ajouté à l'optimiseur, EMA désactivée

Protocole :
  • Initialisation depuis zinc_pretrained.pt (ou best_model.pt en fallback)
  • Fine-tuning sur ChEMBL251 avec chaque variante (--epochs, défaut 50)
  • Évaluation finale : sonde MLP → Pearson r, Spearman ρ, RMSE (test set)
  • 3 runs par variante (seeds 0, 1, 2) pour obtenir mean ± std
  • Résultats dans results/ablation_results.json + tableau ASCII

Usage :
    python ablation_study.py
    python ablation_study.py --epochs 30 --probe-epochs 80 --runs 2
    python ablation_study.py --variants V1 V3          # sous-ensemble de variantes
"""

import argparse
import copy
import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.loader import DataLoader

from data.chembl_dataset import ChEMBLDataset, creer_dataloaders
from evaluation.metrics import SondeMLP, calculer_metriques, extraire_embeddings
from models.bio_jepa import BioJEPA
from training.ema import EMAUpdater


# ── Configuration des variantes ───────────────────────────────────────────────

@dataclass
class ConfigVariante:
    id: str
    nom: str
    no_ema: bool = False         # V2 : copie directe à chaque batch
    fixed_mask: bool = False     # V3 : masquage fixe à 30%
    no_stop_grad: bool = False   # V4 : gradients traversent le Target Encoder

VARIANTES = [
    ConfigVariante('V1', 'Bio-JEPA complet (référence)'),
    ConfigVariante('V2', 'Sans EMA (copie directe τ=0)',     no_ema=True),
    ConfigVariante('V3', 'Sans masquage progressif (30%)',   fixed_mask=True),
    ConfigVariante('V4', 'Sans stop-gradient',               no_stop_grad=True),
]
VARIANTE_PAR_ID = {v.id: v for v in VARIANTES}


# ── Chargement du modèle ──────────────────────────────────────────────────────

def charger_modele(checkpoint_path: str, device: torch.device) -> BioJEPA:
    """Charge un checkpoint Bio-JEPA et retourne le modèle initialisé."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = BioJEPA()
    if isinstance(ckpt, dict) and 'target_encoder' in ckpt:
        model.context_encoder.load_state_dict(ckpt['context_encoder'])
        model.target_encoder.load_state_dict(ckpt['target_encoder'])
        model.predictor.load_state_dict(ckpt['predictor'])
    else:
        model.load_state_dict(ckpt)
    return model.to(device)


# ── Forwards variants ─────────────────────────────────────────────────────────

def forward_standard(model: BioJEPA, data) -> torch.Tensor:
    """Forward standard — identique à BioJEPA.forward() mais retourne juste la loss."""
    loss, _, _, _ = model(data)
    return loss


def forward_sans_stopgrad(model: BioJEPA, data) -> torch.Tensor:
    """
    V4 — Forward sans stop-gradient.

    Différences vs forward standard :
      • Pas de torch.no_grad() autour du Target Encoder
      • Pas de .detach() sur z_target dans la loss
    → Les gradients traversent les deux encodeurs simultanément.
    """
    # Target Encoder WITH gradient (pas de torch.no_grad())
    z_target = model.target_encoder(data.x, data.edge_index, data.edge_attr, data.batch)
    z_target = F.normalize(z_target, dim=-1)

    # Context Encoder (graphe masqué)
    data_masque = model._masquer_graphe(data)
    z_context = model.context_encoder(
        data_masque.x, data_masque.edge_index,
        data_masque.edge_attr, data_masque.batch,
    )

    # Predictor
    z_pred = model.predictor(z_context)
    z_pred = F.normalize(z_pred, dim=-1)

    # Loss SANS .detach() → gradient remonte dans z_target
    return F.mse_loss(z_pred, z_target)


# ── Mise à jour du Target Encoder ────────────────────────────────────────────

@torch.no_grad()
def copier_poids_direct(context_encoder: nn.Module, target_encoder: nn.Module):
    """
    V2 — Copie directe des poids (τ=0, pas d'interpolation EMA).
    Copie aussi les buffers BatchNorm comme EMAUpdater.
    """
    for p_c, p_t in zip(context_encoder.parameters(), target_encoder.parameters()):
        p_t.data.copy_(p_c.data)
    for b_c, b_t in zip(context_encoder.buffers(), target_encoder.buffers()):
        b_t.data.copy_(b_c.data)


# ── Scheduler de masquage ─────────────────────────────────────────────────────

def calculer_mask_ratio(
    epoch: int,
    n_epochs: int,
    start: float,
    end: float,
) -> float:
    """Schedule cosinus du mask_ratio (identique à BioJEPATrainer)."""
    t = (epoch - 1) / max(n_epochs - 1, 1)
    return start + (end - start) * (1 - math.cos(math.pi * t)) / 2


# ── Boucle d'entraînement (variante-agnostique) ───────────────────────────────

def entrainer_variante(
    model: BioJEPA,
    train_loader: DataLoader,
    device: torch.device,
    variante: ConfigVariante,
    n_epochs: int = 50,
    lr: float = 1e-4,
    weight_decay: float = 1e-5,
    gradient_clip: float = 1.0,
    mask_ratio_start: float = 0.10,
    mask_ratio_end: float = 0.30,
    ema_start: float = 0.996,
    ema_end: float = 1.0,
    verbose: bool = True,
) -> BioJEPA:
    """
    Entraîne le modèle selon la configuration de la variante.

    Chaque variante modifie exactement UN composant du mécanisme JEPA
    (EMA, masquage progressif, ou stop-gradient) — tout le reste est identique.
    """
    total_steps = n_epochs * len(train_loader)

    # --- Masquage ---
    mask_start = mask_ratio_end if variante.fixed_mask else mask_ratio_start
    mask_end   = mask_ratio_end

    # --- Paramètres à optimiser ---
    params_trainables = (
        list(model.context_encoder.parameters())
        + list(model.predictor.parameters())
    )
    if variante.no_stop_grad:
        # Le Target Encoder reçoit aussi des gradients
        for p in model.target_encoder.parameters():
            p.requires_grad = True
        params_trainables += list(model.target_encoder.parameters())

    optimizer = AdamW(params_trainables, lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=lr * 0.01)

    # --- EMA ---
    use_ema = (not variante.no_ema) and (not variante.no_stop_grad)
    ema = EMAUpdater(ema_start=ema_start, ema_end=ema_end, total_steps=total_steps) if use_ema else None

    forward_fn = forward_sans_stopgrad if variante.no_stop_grad else forward_standard

    # --- Boucle ---
    model.train()
    for epoch in range(1, n_epochs + 1):
        model.mask_ratio = calculer_mask_ratio(epoch, n_epochs, mask_start, mask_end)
        epoch_loss, n_batches = 0.0, 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            loss = forward_fn(model, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params_trainables, gradient_clip)
            optimizer.step()
            scheduler.step()

            # Mise à jour du Target Encoder
            if variante.no_ema:
                copier_poids_direct(model.context_encoder, model.target_encoder)
            elif ema is not None:
                ema.update(model.context_encoder, model.target_encoder)
            # V4 : pas de mise à jour séparée — target reçoit les gradients de la loss

            epoch_loss += loss.item()
            n_batches += 1

        if verbose and epoch % 10 == 0:
            print(f"      Epoch {epoch:>3}/{n_epochs}  "
                  f"loss={epoch_loss / n_batches:.4f}  "
                  f"mask={model.mask_ratio:.2f}")

    # Regeler le Target Encoder si V4 (pour évaluation propre ensuite)
    if variante.no_stop_grad:
        for p in model.target_encoder.parameters():
            p.requires_grad = False

    model.eval()
    return model


# ── Évaluation par sonde MLP ──────────────────────────────────────────────────

def evaluer_probe(
    model: BioJEPA,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    embedding_dim: int = 256,
    hidden_dim: int = 256,
    n_epochs_probe: int = 100,
) -> dict:
    """
    Sonde MLP 2 couches entraînée sur les embeddings figés du Target Encoder.

    Returns:
        dict avec pearson_r, spearman_r, rmse, mae.
    """
    model.eval()
    z_train, y_train = extraire_embeddings(model, train_loader, device)
    z_test,  y_test  = extraire_embeddings(model, test_loader,  device)

    z_tr = torch.tensor(z_train, dtype=torch.float, device=device)
    y_tr = torch.tensor(y_train, dtype=torch.float, device=device)
    z_te = torch.tensor(z_test,  dtype=torch.float, device=device)

    sonde = SondeMLP(embedding_dim, hidden_dim).to(device)
    opt   = torch.optim.Adam(sonde.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = CosineAnnealingLR(opt, T_max=n_epochs_probe, eta_min=1e-5)

    best_loss, best_state = float('inf'), None
    for _ in range(n_epochs_probe):
        sonde.train()
        opt.zero_grad()
        F.mse_loss(sonde(z_tr), y_tr).backward()
        opt.step()
        sched.step()
        with torch.no_grad():
            loss = F.mse_loss(sonde(z_tr), y_tr).item()
        if loss < best_loss:
            best_loss = loss
            best_state = {k: v.clone() for k, v in sonde.state_dict().items()}

    sonde.load_state_dict(best_state)
    sonde.eval()
    with torch.no_grad():
        y_pred = sonde(z_te).cpu().numpy()

    return calculer_metriques(y_test, y_pred)


# ── Run complet pour une variante ─────────────────────────────────────────────

def run_variante(
    variante: ConfigVariante,
    checkpoint_path: str,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    seeds: list[int],
    n_epochs: int,
    n_epochs_probe: int,
    embedding_dim: int = 256,
) -> dict:
    """
    Exécute N runs pour une variante et agrège les métriques.

    Returns:
        dict avec mean/std de pearson_r, spearman_r, rmse sur les N seeds.
    """
    pearson_rs, spearman_rs, rmses = [], [], []

    for run_idx, seed in enumerate(seeds):
        t0 = time.time()
        print(f"\n    [Run {run_idx+1}/{len(seeds)}  seed={seed}]")

        # Seed de reproductibilité
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Chargement frais du checkpoint à chaque run
        model = charger_modele(checkpoint_path, device)

        # Entraînement avec la variante
        model = entrainer_variante(
            model, train_loader, device, variante,
            n_epochs=n_epochs,
        )

        # Évaluation par sonde MLP
        metriques = evaluer_probe(
            model, train_loader, test_loader, device,
            embedding_dim=embedding_dim,
            n_epochs_probe=n_epochs_probe,
        )

        pearson_rs.append(metriques['pearson_r'])
        spearman_rs.append(metriques['spearman_r'])
        rmses.append(metriques['rmse'])

        print(f"      → Pearson r={metriques['pearson_r']:.4f}  "
              f"Spearman ρ={metriques['spearman_r']:.4f}  "
              f"RMSE={metriques['rmse']:.4f}  "
              f"({time.time()-t0:.0f}s)")

        # Libérer la mémoire GPU
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {
        'pearson_r'  : {'mean': float(np.mean(pearson_rs)),  'std': float(np.std(pearson_rs)),  'runs': pearson_rs},
        'spearman_r' : {'mean': float(np.mean(spearman_rs)), 'std': float(np.std(spearman_rs)), 'runs': spearman_rs},
        'rmse'       : {'mean': float(np.mean(rmses)),        'std': float(np.std(rmses)),        'runs': rmses},
    }


# ── Affichage du tableau récapitulatif ────────────────────────────────────────

def afficher_tableau(resultats: dict):
    """Affiche un tableau ASCII des résultats de l'ablation."""
    largeur = 84
    print("\n" + "═" * largeur)
    print("  RÉSULTATS DE L'ÉTUDE D'ABLATION — Bio-JEPA sur CHEMBL251")
    print("═" * largeur)
    print(f"  {'Variante':<36}  {'Pearson r':>14}  {'Spearman ρ':>14}  {'RMSE':>11}")
    print("  " + "─" * (largeur - 2))

    # Référence pour le delta
    ref_r = resultats.get('V1', {}).get('pearson_r', {}).get('mean', None)

    for vid, res in resultats.items():
        nom = VARIANTE_PAR_ID[vid].nom
        nom_court = nom[:35]
        pr = res['pearson_r']
        sr = res['spearman_r']
        rm = res['rmse']

        delta = ''
        if ref_r is not None and vid != 'V1':
            diff = pr['mean'] - ref_r
            delta = f" ({diff:+.3f})"

        print(
            f"  {nom_court:<36}"
            f"  {pr['mean']:.3f} ± {pr['std']:.3f}{delta:>9}"
            f"  {sr['mean']:.3f} ± {sr['std']:.3f}"
            f"  {rm['mean']:.3f} ± {rm['std']:.3f}"
        )

    print("═" * largeur)
    print(f"  (+/-) = delta Pearson r vs V1 (référence)\n")


# ── Point d'entrée ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Étude d'ablation Bio-JEPA sur ChEMBL251",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        '--checkpoint', default='checkpoints/zinc_pretrained.pt',
        help='Checkpoint de départ pour toutes les variantes',
    )
    parser.add_argument(
        '--target', default='CHEMBL251',
        help='Cible ChEMBL (défaut: CHEMBL251)',
    )
    parser.add_argument(
        '--epochs', type=int, default=50,
        help="Époques de fine-tuning par variante (défaut: 50)",
    )
    parser.add_argument(
        '--probe-epochs', type=int, default=100,
        help="Époques d'entraînement de la sonde MLP (défaut: 100)",
    )
    parser.add_argument(
        '--runs', type=int, default=3,
        help='Nombre de runs par variante (défaut: 3)',
    )
    parser.add_argument(
        '--variants', nargs='+', choices=['V1', 'V2', 'V3', 'V4'],
        default=['V1', 'V2', 'V3', 'V4'],
        help='Sous-ensemble de variantes à tester',
    )
    parser.add_argument(
        '--batch-size', type=int, default=32,
    )
    parser.add_argument(
        '--out-dir', default='results',
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

    # --- Checkpoint de départ ---
    if not os.path.exists(args.checkpoint):
        fallback = 'checkpoints/best_model.pt'
        if os.path.exists(fallback):
            print(f"  {args.checkpoint} absent → fallback {fallback}")
            args.checkpoint = fallback
        else:
            raise FileNotFoundError(
                f"Aucun checkpoint trouvé.\n"
                "Lancez d'abord :  python pretrain.py  ou  python train.py --target CHEMBL251"
            )
    print(f"[Checkpoint] {args.checkpoint}")

    # --- Dataset ChEMBL251 ---
    print(f"\n[Données] ChEMBL — target={args.target}")
    dataset = ChEMBLDataset(root='data/chembl', target_chembl_id=args.target)
    train_loader, _, test_loader = creer_dataloaders(
        dataset, batch_size=args.batch_size, seed=42,
    )
    print(f"  Train : {len(train_loader.dataset):,}  |  Test : {len(test_loader.dataset):,}")

    # --- Seeds ---
    seeds = list(range(args.runs))

    # --- Estimation du temps ---
    n_variantes = len(args.variants)
    print(f"\n[Plan] {n_variantes} variantes × {args.runs} runs × {args.epochs} epochs"
          f" + {args.probe_epochs} epochs sonde")
    print(f"       Estimé : ~{n_variantes * args.runs * (args.epochs * 15 + args.probe_epochs * 2) // 60} min "
          f"(approximatif, CPU/GPU)")

    # ─── Boucle principale ───────────────────────────────────────────────────
    resultats = {}
    t_total = time.time()

    for vid in args.variants:
        variante = VARIANTE_PAR_ID[vid]
        print(f"\n{'─'*60}")
        print(f"  {vid} — {variante.nom}")
        print(f"{'─'*60}")
        t_var = time.time()

        res = run_variante(
            variante=variante,
            checkpoint_path=args.checkpoint,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            seeds=seeds,
            n_epochs=args.epochs,
            n_epochs_probe=args.probe_epochs,
        )
        resultats[vid] = res

        elapsed = time.time() - t_var
        print(f"\n  {vid} terminé en {elapsed:.0f}s  "
              f"→ Pearson r = {res['pearson_r']['mean']:.3f} ± {res['pearson_r']['std']:.3f}")

    print(f"\n[Temps total] {(time.time()-t_total)/60:.1f} min")

    # ─── Tableau récapitulatif ───────────────────────────────────────────────
    afficher_tableau(resultats)

    # ─── Sauvegarde JSON ────────────────────────────────────────────────────
    output = {
        'metadata': {
            'checkpoint'    : args.checkpoint,
            'target'        : args.target,
            'epochs_finetune': args.epochs,
            'epochs_probe'  : args.probe_epochs,
            'n_runs'        : args.runs,
            'seeds'         : seeds,
            'device'        : str(device),
        },
        'variantes': {
            vid: {
                'nom'       : VARIANTE_PAR_ID[vid].nom,
                'metriques' : res,
            }
            for vid, res in resultats.items()
        },
    }

    out_path = os.path.join(args.out_dir, 'ablation_results.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"[Résultats] Sauvegardés → {out_path}")


if __name__ == '__main__':
    main()
