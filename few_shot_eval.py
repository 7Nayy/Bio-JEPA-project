"""
few_shot_eval.py

Évaluation few-shot : Bio-JEPA vs GNN supervisé from scratch.

Motivation : en régime full-data, le GNN supervisé peut dominer Bio-JEPA
car il dispose d'un signal de supervision direct sur toutes les molécules.
La vraie contribution de JEPA est le transfert en régime few-shot :
  → les features pré-entraînées sur la structure moléculaire sans labels
    devraient être bien meilleures que les features apprises from scratch
    sur seulement 10, 50 ou 100 exemples.

Protocole expérimental :
──────────────────────────────────────────────────────────────────────────
  Pour chaque N ∈ {10, 50, 100, 200, 500, 1000} :
    Bio-JEPA    → charge le checkpoint (Target Encoder figé)
                  → extrait les embeddings de tout le train set
                  → sous-échantillonne N exemples
                  → entraîne SondeMLP sur N embeddings (200 époques)
                  → évalue sur le test set complet

    GNN supervisé → réinstancie le GNNRegresseur from scratch (seed aléatoire)
                    → sous-échantillonne N exemples du train set
                    → entraîne end-to-end 300 époques sur N exemples
                    → early stopping sur val_loss (même val set)
                    → évalue sur le test set complet

  n_runs tirages différents de N → moyenne ± std (robustesse statistique).

Garanties de comparaison équitable :
  • Même split 80/10/10 seed=42 pour les deux modèles.
  • Même N exemples par run (seed 42+run pour le sous-échantillonnage).
  • Même val_loader pour l'early stopping des deux modèles.
  • Même test_loader pour l'évaluation finale.
  • Mêmes métriques : Pearson r, Spearman ρ, RMSE.

Usage :
    # Évaluation sur EGFR (CHEMBL203) avec le checkpoint ZINC
    python few_shot_eval.py --checkpoint checkpoints/zinc_pretrained.pt \\
        --target CHEMBL203 --n-values 10,50,100,200,500,1000 \\
        --n-runs 5 --save-json results/few_shot_egfr.json

    # Évaluation sur A2A (CHEMBL251) avec le meilleur modèle
    python few_shot_eval.py --checkpoint checkpoints/best_model.pt \\
        --target CHEMBL251 --save-json results/few_shot.json

    # Dataset démonstration (test du pipeline)
    python few_shot_eval.py --checkpoint checkpoints/best_model.pt --demo
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from scipy import stats
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader

from baseline import GNNRegresseur
from data.chembl_dataset import ChEMBLDataset, creer_dataloaders, creer_dataset_demo
from evaluation.metrics import SondeMLP
from models.bio_jepa import BioJEPA


# ── Noms de cibles connus (pour l'affichage) ──────────────────────────────────

NOMS_CIBLES = {
    'CHEMBL251': 'A2A (adénosine)',
    'CHEMBL203': 'EGFR (kinase)',
    'CHEMBL205': 'ERB2 (HER2)',
    'CHEMBL301': 'ACE (enzyme)',
    'CHEMBL240': 'hERG (canal K+)',
}


def nom_cible(target_id: str) -> str:
    return NOMS_CIBLES.get(target_id, target_id)


# ---------------------------------------------------------------------------
# Chargement du checkpoint Bio-JEPA
# ---------------------------------------------------------------------------

def charger_biojep(chemin: str, cfg_m: dict, device) -> BioJEPA:
    """
    Reconstruit et charge un modèle BioJEPA depuis un checkpoint sauvegardé
    par BioJEPATrainer.

    Args:
        chemin  : Chemin vers le fichier .pt (ex: 'checkpoints/best_model.pt').
        cfg_m   : Sous-dictionnaire config['modele'].
        device  : Dispositif cible.

    Returns:
        BioJEPA en mode eval, prêt pour l'extraction d'embeddings.
    """
    modele = BioJEPA(
        hidden_dim=cfg_m['hidden_dim'],
        embedding_dim=cfg_m['embedding_dim'],
        num_gnn_layers=cfg_m['num_gnn_layers'],
        predictor_hidden_dim=cfg_m['predictor_hidden_dim'],
        predictor_num_layers=cfg_m['predictor_num_layers'],
        mask_ratio=cfg_m['mask_ratio_start'],
        dropout=cfg_m['dropout'],
    )
    ckpt = torch.load(chemin, map_location=device, weights_only=False)
    modele.context_encoder.load_state_dict(ckpt['context_encoder'])
    modele.target_encoder.load_state_dict(ckpt['target_encoder'])
    modele.predictor.load_state_dict(ckpt['predictor'])
    modele = modele.to(device)
    modele.eval()
    return modele


def resoudre_checkpoint(chemin: str) -> str:
    """
    Résout le chemin du checkpoint avec fallback automatique.

    Priorité :
      1. Chemin fourni
      2. checkpoints/zinc_pretrained.pt
      3. checkpoints/best_model.pt
    """
    if os.path.exists(chemin):
        return chemin
    for fallback in ('checkpoints/zinc_pretrained.pt', 'checkpoints/best_model.pt'):
        if os.path.exists(fallback) and fallback != chemin:
            print(f"  Checkpoint '{chemin}' absent → fallback : {fallback}")
            return fallback
    raise FileNotFoundError(
        f"Aucun checkpoint trouvé (cherché : {chemin}).\n"
        "Lancez d'abord : python train.py --target CHEMBL251"
    )


# ---------------------------------------------------------------------------
# Extraction des embeddings Bio-JEPA
# ---------------------------------------------------------------------------

@torch.no_grad()
def extraire_embeddings_liste(
    model: BioJEPA,
    data_list: list,
    device,
    batch_size: int = 256,
) -> tuple:
    """
    Extrait les embeddings du Target Encoder pour une liste de Data PyG.

    L'extraction est faite en mini-batches pour éviter les OOM sur CPU/GPU.
    Les embeddings sont normalisés L2 (idem BioJEPA.encode).

    Args:
        model     : BioJEPA en mode eval.
        data_list : Liste de Data PyG.
        device    : Dispositif.
        batch_size: Taille des mini-batches d'extraction.

    Returns:
        (Z, y) — tableaux numpy [N, D] et [N].
    """
    model.eval()
    Z_list, y_list = [], []

    for start in range(0, len(data_list), batch_size):
        sous_liste = data_list[start:start + batch_size]
        batch = Batch.from_data_list(sous_liste).to(device)
        z = model.encode(batch, encoder='target')
        Z_list.append(z.cpu().numpy())
        y_list.append(batch.y.cpu().numpy().flatten())

    return np.vstack(Z_list), np.concatenate(y_list)


@torch.no_grad()
def extraire_embeddings_loader(model: BioJEPA, loader, device) -> tuple:
    """
    Extrait les embeddings du Target Encoder depuis un DataLoader PyG.

    Returns:
        (Z, y) — tableaux numpy [N, D] et [N].
    """
    model.eval()
    Z_list, y_list = [], []

    for batch in loader:
        batch = batch.to(device)
        z = model.encode(batch, encoder='target')
        Z_list.append(z.cpu().numpy())
        y_list.append(batch.y.cpu().numpy().flatten())

    return np.vstack(Z_list), np.concatenate(y_list)


# ---------------------------------------------------------------------------
# Primitives d'entraînement silencieuses (sans I/O disque ni print)
# ---------------------------------------------------------------------------

def _entrainer_sonde(
    Z_train: np.ndarray,
    y_train: np.ndarray,
    Z_val: np.ndarray,
    y_val: np.ndarray,
    Z_test: np.ndarray,
    y_test: np.ndarray,
    embedding_dim: int,
    hidden_dim: int,
    lr: float,
    num_epochs: int,
    device,
) -> dict:
    """
    Entraîne une SondeMLP sur N embeddings figés et retourne les métriques
    sur le test set.

    Early stopping sur val_MSE : garde les meilleurs poids en mémoire.
    Pas de sauvegarde disque → adapté à l'exécution de centaines d'expériences.

    Args:
        Z_train, y_train : Embeddings et labels train [N, D] et [N].
        Z_val, y_val     : Embeddings et labels val (early stopping).
        Z_test, y_test   : Embeddings et labels test (évaluation finale).
        embedding_dim    : Dimension des embeddings D.
        hidden_dim       : Dimension cachée de la SondeMLP.
        lr               : Learning rate initial.
        num_epochs       : Nombre max d'époques.
        device           : Dispositif.

    Returns:
        dict avec 'pearson_r', 'spearman_r', 'rmse'.
    """
    def _t(arr):
        return torch.tensor(arr, dtype=torch.float, device=device)

    Z_tr, y_tr = _t(Z_train), _t(y_train)
    Z_vl, y_vl = _t(Z_val), _t(y_val)
    Z_te = _t(Z_test)

    sonde = SondeMLP(embedding_dim, hidden_dim).to(device)
    optim = torch.optim.Adam(sonde.parameters(), lr=lr, weight_decay=1e-4)
    sched = CosineAnnealingLR(optim, T_max=num_epochs, eta_min=lr * 0.01)

    meilleure_val = float('inf')
    params_best = {k: v.clone() for k, v in sonde.state_dict().items()}

    for _ in range(num_epochs):
        sonde.train()
        optim.zero_grad()
        F.mse_loss(sonde(Z_tr), y_tr).backward()
        optim.step()
        sched.step()

        sonde.eval()
        with torch.no_grad():
            val_mse = F.mse_loss(sonde(Z_vl), y_vl).item()

        if val_mse < meilleure_val:
            meilleure_val = val_mse
            params_best = {k: v.clone() for k, v in sonde.state_dict().items()}

    sonde.load_state_dict(params_best)
    sonde.eval()
    with torch.no_grad():
        pred = sonde(Z_te).cpu().numpy()

    pearson_r, _ = stats.pearsonr(y_test, pred)
    spearman_r, _ = stats.spearmanr(y_test, pred)
    rmse = float(np.sqrt(np.mean((y_test - pred) ** 2)))

    return {
        'pearson_r' : float(pearson_r),
        'spearman_r': float(spearman_r),
        'rmse'      : rmse,
    }


def _entrainer_gnn_fewshot(
    model: GNNRegresseur,
    train_loader,
    val_loader,
    num_epochs: int,
    lr: float,
    weight_decay: float,
    gradient_clip: float,
    device,
) -> GNNRegresseur:
    """
    Entraîne un GNNRegresseur from scratch en mode silencieux.

    Early stopping sur val_MSE, best weights gardés en mémoire.
    Identique à entrainer_baseline() mais sans print ni sauvegarde disque.

    Returns:
        GNNRegresseur avec les meilleurs poids (val_MSE minimal).
    """
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = num_epochs * len(train_loader)
    sched = CosineAnnealingLR(
        optimizer, T_max=max(total_steps, 1), eta_min=lr * 0.01
    )

    meilleure_val = float('inf')
    params_best = {k: v.clone() for k, v in model.state_dict().items()}

    for _ in range(num_epochs):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            F.mse_loss(model(batch), batch.y.view(-1)).backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=gradient_clip
            )
            optimizer.step()
            sched.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                val_losses.append(
                    F.mse_loss(model(batch), batch.y.view(-1)).item()
                )
        val_loss = sum(val_losses) / len(val_losses)

        if val_loss < meilleure_val:
            meilleure_val = val_loss
            params_best = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(params_best)
    return model


# ---------------------------------------------------------------------------
# Expériences few-shot
# ---------------------------------------------------------------------------

def experience_fewshot_biojep(
    Z_all_train: np.ndarray,
    y_all_train: np.ndarray,
    Z_val: np.ndarray,
    y_val: np.ndarray,
    Z_test: np.ndarray,
    y_test: np.ndarray,
    N: int,
    n_runs: int,
    embedding_dim: int,
    hidden_dim: int,
    lr: float,
    num_epochs_sonde: int,
    device,
    verbose: bool = True,
) -> dict:
    """
    Évalue Bio-JEPA en few-shot sur N exemples labellisés.

    Les embeddings sont pré-extraits une fois → chaque run ne fait
    qu'un sous-échantillonnage + entraînement de SondeMLP (très rapide).

    Returns:
        dict {metric: {'mean': float, 'std': float, 'runs': list}}
        pour pearson_r, spearman_r, rmse.
    """
    N_eff = min(N, len(Z_all_train))
    pearson_rs, spearman_rs, rmses = [], [], []

    for run in range(n_runs):
        rng = np.random.default_rng(seed=42 + run)
        # Même tirage que GNN (même seed → même N exemples)
        idx = rng.choice(len(Z_all_train), size=N_eff, replace=False)

        metriques = _entrainer_sonde(
            Z_train=Z_all_train[idx],
            y_train=y_all_train[idx],
            Z_val=Z_val,
            y_val=y_val,
            Z_test=Z_test,
            y_test=y_test,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            lr=lr,
            num_epochs=num_epochs_sonde,
            device=device,
        )
        pearson_rs.append(metriques['pearson_r'])
        spearman_rs.append(metriques['spearman_r'])
        rmses.append(metriques['rmse'])

        if verbose:
            print(
                f"    Bio-JEPA  N={N_eff:>5}  run {run+1}/{n_runs}"
                f"  →  r={metriques['pearson_r']:+.4f}"
                f"  ρ={metriques['spearman_r']:+.4f}"
                f"  RMSE={metriques['rmse']:.4f}"
            )

    return {
        'pearson_r' : {'mean': float(np.mean(pearson_rs)),  'std': float(np.std(pearson_rs)),  'runs': pearson_rs},
        'spearman_r': {'mean': float(np.mean(spearman_rs)), 'std': float(np.std(spearman_rs)), 'runs': spearman_rs},
        'rmse'      : {'mean': float(np.mean(rmses)),        'std': float(np.std(rmses)),        'runs': rmses},
    }


def experience_fewshot_gnn(
    all_train_data: list,
    val_loader,
    test_loader,
    N: int,
    n_runs: int,
    cfg_m: dict,
    cfg_t: dict,
    num_epochs: int,
    device,
    verbose: bool = True,
) -> dict:
    """
    Évalue le GNN supervisé en few-shot sur N exemples labellisés.

    Chaque run instancie un GNNRegresseur with fresh random weights,
    l'entraîne from scratch sur N exemples, et évalue sur le test set complet.

    Returns:
        dict {metric: {'mean': float, 'std': float, 'runs': list}}
        pour pearson_r, spearman_r, rmse.
    """
    N_eff = min(N, len(all_train_data))
    pearson_rs, spearman_rs, rmses = [], [], []

    for run in range(n_runs):
        # Même tirage que Bio-JEPA pour ce run (même seed)
        rng = np.random.default_rng(seed=42 + run)
        idx = rng.choice(len(all_train_data), size=N_eff, replace=False)
        sous_train = [all_train_data[i] for i in idx]

        # Réinitialisation des poids du GNN (reproductible)
        torch.manual_seed(1000 + run)  # seed différent de Bio-JEPA mais fixe
        if torch.cuda.is_available():
            torch.cuda.manual_seed(1000 + run)

        model = GNNRegresseur(
            hidden_dim=cfg_m['hidden_dim'],
            embedding_dim=cfg_m['embedding_dim'],
            num_layers=cfg_m['num_gnn_layers'],
            mlp_hidden=cfg_m['embedding_dim'],
            dropout=cfg_m['dropout'],
        ).to(device)

        # Taille de batch adaptée à N (BatchNorm nécessite >= 2 exemples/batch)
        batch_size = min(N_eff, 32)
        # drop_last si le dernier batch serait de taille 1 (crash BatchNorm)
        drop_last = (N_eff > batch_size) and (N_eff % batch_size == 1)

        train_loader_N = DataLoader(
            sous_train, batch_size=batch_size, shuffle=True, drop_last=drop_last
        )

        model = _entrainer_gnn_fewshot(
            model=model,
            train_loader=train_loader_N,
            val_loader=val_loader,
            num_epochs=num_epochs,
            lr=cfg_t['lr'],
            weight_decay=cfg_t['weight_decay'],
            gradient_clip=cfg_t['gradient_clip'],
            device=device,
        )

        # Évaluation sur le test set complet
        model.eval()
        y_vrai_list, y_pred_list = [], []
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                y_vrai_list.append(batch.y.view(-1).cpu().numpy())
                y_pred_list.append(model(batch).cpu().numpy())

        y_vrai = np.concatenate(y_vrai_list)
        y_pred = np.concatenate(y_pred_list)

        pearson_r, _ = stats.pearsonr(y_vrai, y_pred)
        spearman_r, _ = stats.spearmanr(y_vrai, y_pred)
        rmse = float(np.sqrt(np.mean((y_vrai - y_pred) ** 2)))

        pearson_rs.append(float(pearson_r))
        spearman_rs.append(float(spearman_r))
        rmses.append(rmse)

        if verbose:
            print(
                f"    GNN sup.  N={N_eff:>5}  run {run+1}/{n_runs}"
                f"  →  r={pearson_r:+.4f}"
                f"  ρ={spearman_r:+.4f}"
                f"  RMSE={rmse:.4f}"
            )

    return {
        'pearson_r' : {'mean': float(np.mean(pearson_rs)),  'std': float(np.std(pearson_rs)),  'runs': pearson_rs},
        'spearman_r': {'mean': float(np.mean(spearman_rs)), 'std': float(np.std(spearman_rs)), 'runs': spearman_rs},
        'rmse'      : {'mean': float(np.mean(rmses)),        'std': float(np.std(rmses)),        'runs': rmses},
    }


# ---------------------------------------------------------------------------
# Visualisation ASCII
# ---------------------------------------------------------------------------

def afficher_courbe_ascii(
    resultats_jepa: list,
    resultats_gnn: list,
    N_values: list,
    n_runs: int,
    target_id: str = 'CHEMBL251',
) -> None:
    """
    Affiche une courbe ASCII 2D : Pearson r vs N pour les deux modèles.

    Args:
        resultats_jepa : [dict_metriques, ...] pour chaque N (issu de experience_fewshot_biojep).
        resultats_gnn  : [dict_metriques, ...] pour chaque N.
        N_values       : Liste des valeurs de N.
        n_runs         : Nombre de runs (pour l'annotation des barres d'erreur).
        target_id      : Identifiant de la cible ChEMBL.
    """
    # Extraction des (mean_r, std_r) pour le tracé
    def extraire(res_list):
        return [(r['pearson_r']['mean'], r['pearson_r']['std']) for r in res_list]

    vals_jepa = extraire(resultats_jepa)
    vals_gnn  = extraire(resultats_gnn)

    HAUTEUR = 16       # résolution verticale (Y)
    PAS_X = 10         # colonnes par valeur de N
    Y_MIN, Y_MAX = 0.0, 1.0

    largeur = len(N_values) * PAS_X
    grille = [[' '] * largeur for _ in range(HAUTEUR)]

    def r_to_row(r: float) -> int:
        r = max(Y_MIN, min(Y_MAX, float(r)))
        return int(round((1.0 - r) / (Y_MAX - Y_MIN) * (HAUTEUR - 1)))

    def n_to_col(i: int) -> int:
        return i * PAS_X + PAS_X // 2

    def mettre(row: int, col: int, char: str, force: bool = False):
        if 0 <= row < HAUTEUR and 0 <= col < largeur:
            if force or grille[row][col] == ' ':
                grille[row][col] = char

    def tracer_segment(r1: int, c1: int, r2: int, c2: int, sym: str):
        if c2 <= c1 + 1:
            return
        for c in range(c1 + 1, c2):
            t = (c - c1) / (c2 - c1)
            r = int(round(r1 + t * (r2 - r1)))
            mettre(r, c, sym, force=False)

    for vals, sym_pt, sym_ln in [
        (vals_jepa, '●', '─'),
        (vals_gnn,  '○', '╌'),
    ]:
        points = []
        for i, (mean_r, std_r) in enumerate(vals):
            if not np.isnan(mean_r):
                row = r_to_row(mean_r)
                col = n_to_col(i)
                points.append((row, col, std_r))

        for k in range(len(points) - 1):
            r1, c1, _ = points[k]
            r2, c2, _ = points[k + 1]
            tracer_segment(r1, c1, r2, c2, sym_ln)

        rows_per_unit = (HAUTEUR - 1) / (Y_MAX - Y_MIN)
        for row, col, std_r in points:
            delta_rows = int(round(std_r * rows_per_unit))
            for dr in range(1, delta_rows + 1):
                mettre(row - dr, col, '│')
                mettre(row + dr, col, '│')

        for row, col, _ in points:
            mettre(row, col, sym_pt, force=True)

    cible_label = f"{target_id} — {nom_cible(target_id)}"
    print(f"\n  Pearson r (test set) vs N exemples labellisés — {cible_label}")
    print(f"  Barres d'erreur = ±1 std sur {n_runs} sous-échantillonnages aléatoires")
    print()

    for i, ligne in enumerate(grille):
        r_val = Y_MAX - (Y_MAX - Y_MIN) * i / (HAUTEUR - 1)
        tick = f"{r_val:.1f}" if i % 4 == 0 else "   "
        print(f"  {tick} │{''.join(ligne)}")

    print(f"       └{'─' * largeur}")

    label_row = [' '] * largeur
    for i, N in enumerate(N_values):
        col = n_to_col(i)
        label = str(N)
        start = col - len(label) // 2
        for j, ch in enumerate(label):
            pos = start + j
            if 0 <= pos < largeur:
                label_row[pos] = ch
    print(f"        {''.join(label_row)}")
    print(f"           {'← N exemples labellisés →':^{largeur}}")

    print()
    print(f"  Légende : ●─── Bio-JEPA  (Target Encoder figé + SondeMLP)")
    print(f"            ○╌╌╌ GNN supervisé (entraîné from scratch sur N exemples)")
    print()


def afficher_tableau(
    resultats_jepa: list,
    resultats_gnn: list,
    N_values: list,
    n_runs: int,
    target_id: str = 'CHEMBL251',
) -> None:
    """
    Affiche un tableau comparatif avec Pearson r, Spearman ρ et RMSE pour chaque N.

    Le ✓ marque le meilleur modèle (Pearson r) pour chaque N.
    La ligne Δ = Bio-JEPA − GNN indique l'avantage absolu de JEPA.

    Args:
        resultats_jepa : [dict_metriques, ...] par N.
        resultats_gnn  : [dict_metriques, ...] par N.
        N_values       : Liste des valeurs de N.
        n_runs         : Utilisé pour l'annotation.
        target_id      : Identifiant de la cible ChEMBL.
    """
    COL_W = 14

    sep = f"  {'─'*10}─┼─" + '─┼─'.join(['─'*COL_W] * len(N_values)) + '─'
    header = f"  {'N →':>10} │ "
    header += ' │ '.join(f"{N:^{COL_W}}" for N in N_values)

    cible_label = f"{target_id} — {nom_cible(target_id)}"
    print()
    print(f"  {'─'*70}")
    print(f"  Pearson r ± std ({n_runs} runs) — Test set — {cible_label}")
    print(f"  {'─'*70}")
    print(header)
    print(sep)

    def formater(mean_r, std_r, gagnant):
        s = f"{mean_r:+.3f}±{std_r:.3f}"
        return f"{s} ✓" if gagnant else f"{s}  "

    ligne_jepa  = f"  {'Bio-JEPA r':>10} │ "
    ligne_gnn   = f"  {'GNN sup. r':>10} │ "
    ligne_delta = f"  {'Δ (J−G)':>10} │ "

    for res_j, res_g in zip(resultats_jepa, resultats_gnn):
        mj = res_j['pearson_r']['mean']
        sj = res_j['pearson_r']['std']
        mg = res_g['pearson_r']['mean']
        sg = res_g['pearson_r']['std']
        jepa_gagne = mj >= mg
        col_j = formater(mj, sj, jepa_gagne)
        col_g = formater(mg, sg, not jepa_gagne)
        delta = mj - mg
        signe = '+' if delta >= 0 else ''
        col_d = f"{signe}{delta:.3f}"
        ligne_jepa  += f"{col_j:^{COL_W+2}} │ "
        ligne_gnn   += f"{col_g:^{COL_W+2}} │ "
        ligne_delta += f"{col_d:^{COL_W+2}} │ "

    print(ligne_jepa.rstrip(' │ '))
    print(ligne_gnn.rstrip(' │ '))
    print(sep)
    print(ligne_delta.rstrip(' │ '))

    # Spearman ρ
    print()
    print(f"  {'─'*70}")
    print(f"  Spearman ρ ± std ({n_runs} runs) — Test set — {cible_label}")
    print(f"  {'─'*70}")
    print(header)
    print(sep)

    ligne_jepa_s = f"  {'Bio-JEPA ρ':>10} │ "
    ligne_gnn_s  = f"  {'GNN sup. ρ':>10} │ "
    for res_j, res_g in zip(resultats_jepa, resultats_gnn):
        mj = res_j['spearman_r']['mean'];  sj = res_j['spearman_r']['std']
        mg = res_g['spearman_r']['mean'];  sg = res_g['spearman_r']['std']
        pad = ' ' * max(0, COL_W - 12)
        ligne_jepa_s += f"{mj:+.3f}±{sj:.3f}  {pad} │ "
        ligne_gnn_s  += f"{mg:+.3f}±{sg:.3f}  {pad} │ "
    print(ligne_jepa_s.rstrip(' │ '))
    print(ligne_gnn_s.rstrip(' │ '))
    print(sep)

    # RMSE
    print()
    print(f"  {'─'*70}")
    print(f"  RMSE ± std ({n_runs} runs) — Test set — {cible_label}")
    print(f"  {'─'*70}")
    print(header)
    print(sep)

    ligne_jepa_rmse = f"  {'Bio-JEPA':>10} │ "
    ligne_gnn_rmse  = f"  {'GNN sup.':>10} │ "
    for res_j, res_g in zip(resultats_jepa, resultats_gnn):
        mj = res_j['rmse']['mean'];  sj = res_j['rmse']['std']
        mg = res_g['rmse']['mean'];  sg = res_g['rmse']['std']
        pad = ' ' * max(0, COL_W - 12)
        ligne_jepa_rmse += f"{mj:.3f}±{sj:.3f}  {pad} │ "
        ligne_gnn_rmse  += f"{mg:.3f}±{sg:.3f}  {pad} │ "
    print(ligne_jepa_rmse.rstrip(' │ '))
    print(ligne_gnn_rmse.rstrip(' │ '))
    print(f"  {'─'*70}")

    # Seuil de croisement (basé sur Pearson r)
    croisements = [
        N_values[i] for i, (res_j, res_g) in enumerate(zip(resultats_jepa, resultats_gnn))
        if res_j['pearson_r']['mean'] >= res_g['pearson_r']['mean']
    ]
    if croisements:
        print(
            f"\n  Bio-JEPA ≥ GNN pour N ≤ {max(croisements)}"
            f" — avantage few-shot confirmé jusqu'à {max(croisements)} exemples."
        )
    else:
        print(
            "\n  GNN supervisé domine sur tous les N testés.\n"
            "  Considérer un entraînement plus long de Bio-JEPA ou un dataset plus grand."
        )
    print()


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------

def charger_config(chemin: str) -> dict:
    with open(chemin) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description='Évaluation few-shot : Bio-JEPA vs GNN supervisé',
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        '--checkpoint',
        default='checkpoints/zinc_pretrained.pt',
        help='Checkpoint Bio-JEPA (défaut: checkpoints/zinc_pretrained.pt)',
    )
    parser.add_argument(
        '--config',
        default='configs/default.yaml',
        help='Fichier de configuration (défaut: configs/default.yaml)',
    )
    parser.add_argument(
        '--demo', action='store_true',
        help='Dataset synthétique (test rapide du pipeline)',
    )
    parser.add_argument(
        '--csv', type=str, default=None,
        help="CSV local avec colonnes 'smiles' et 'pchembl_value'",
    )
    parser.add_argument(
        '--target', type=str, default='CHEMBL251',
        help="ID de cible ChEMBL (défaut: CHEMBL251 — A2A)\n"
             "Exemples : CHEMBL203 (EGFR), CHEMBL205 (HER2), CHEMBL240 (hERG)",
    )
    parser.add_argument(
        '--n-values', type=str, default='10,50,100,200,500,1000',
        help='Valeurs de N séparées par des virgules (défaut: 10,50,100,200,500,1000)',
    )
    parser.add_argument(
        '--n-runs', type=int, default=5,
        help='Nombre de sous-échantillonnages aléatoires par N (défaut: 5)',
    )
    parser.add_argument(
        '--epochs-supervised', type=int, default=300,
        help='Époques pour le GNN supervisé few-shot (défaut: 300)',
    )
    parser.add_argument(
        '--epochs-probe', type=int, default=200,
        help='Époques pour la SondeMLP Bio-JEPA few-shot (défaut: 200)',
    )
    parser.add_argument(
        '--save-json', type=str, default=None,
        help='Sauvegarder les résultats en JSON (ex: results/few_shot_egfr.json)',
    )
    args = parser.parse_args()

    # ── Configuration ──────────────────────────────────────────────────────
    config = charger_config(args.config)
    cfg_m = config['modele']
    cfg_t = config['entrainement']
    cfg_d = config['donnees']
    cfg_e = config['evaluation']

    N_values = [int(n) for n in args.n_values.split(',')]
    n_runs = args.n_runs
    target_id = args.target

    # ── Dispositif ─────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    cible_label = f"{target_id} — {nom_cible(target_id)}"
    print("=" * 70)
    print(f"  Évaluation Few-Shot : Bio-JEPA vs GNN supervisé")
    print(f"  Cible : {cible_label}")
    print(f"  N ∈ {N_values}")
    print(f"  {n_runs} runs par N  │  Dispositif : {device}")
    print("=" * 70)

    # ── Données — identiques à train.py et baseline.py ────────────────────
    if args.demo:
        print("\n[Données] Mode démonstration")
        data_list = creer_dataset_demo(n_molecules=500)
        n = len(data_list)
        n_train = int(n * cfg_d['train_ratio'])
        n_val = int(n * cfg_d['val_ratio'])

        torch.manual_seed(cfg_d['seed'])
        indices = torch.randperm(n).tolist()

        bs = cfg_t['batch_size']
        train_data = [data_list[i] for i in indices[:n_train]]
        val_data   = [data_list[i] for i in indices[n_train:n_train + n_val]]
        test_data  = [data_list[i] for i in indices[n_train + n_val:]]

        all_train_data = train_data
        val_loader  = DataLoader(val_data, batch_size=bs, shuffle=False)
        test_loader = DataLoader(test_data, batch_size=bs, shuffle=False)
    else:
        print(f"\n[Données] Chargement du dataset ChEMBL — target={target_id}")
        dataset = ChEMBLDataset(
            root=cfg_d['root'],
            csv_path=args.csv,
            target_chembl_id=target_id,
        )
        train_loader, val_loader, test_loader = creer_dataloaders(
            dataset,
            train_ratio=cfg_d['train_ratio'],
            val_ratio=cfg_d['val_ratio'],
            batch_size=cfg_t['batch_size'],
            seed=cfg_d['seed'],
        )
        print("  Matérialisation du train set en liste...")
        train_set = train_loader.dataset
        all_train_data = [train_set[i] for i in range(len(train_set))]

    n_train = len(all_train_data)
    n_val   = len(val_loader.dataset)
    n_test  = len(test_loader.dataset)
    print(
        f"\n  Train total : {n_train:,}  │  Val : {n_val:,}  │  Test : {n_test:,}"
    )

    # Filtrer les N > taille du train set
    N_values_eff = [N for N in N_values if N <= n_train]
    ignores = [N for N in N_values if N > n_train]
    if ignores:
        print(f"  ⚠ N_values ignorés (> train set) : {ignores}")
    N_values = N_values_eff

    # ── Chargement du modèle Bio-JEPA ──────────────────────────────────────
    checkpoint_path = resoudre_checkpoint(args.checkpoint)
    print(f"\n[Bio-JEPA] Checkpoint : {checkpoint_path}")
    model_jepa = charger_biojep(checkpoint_path, cfg_m, device)

    # ── Extraction des embeddings Bio-JEPA (une seule fois) ───────────────
    print("\n[Bio-JEPA] Extraction des embeddings (Target Encoder figé)...")
    t0 = time.time()
    Z_all_train, y_all_train = extraire_embeddings_liste(
        model_jepa, all_train_data, device
    )
    Z_val,  y_val  = extraire_embeddings_loader(model_jepa, val_loader,  device)
    Z_test, y_test = extraire_embeddings_loader(model_jepa, test_loader, device)
    print(
        f"  Train : {Z_all_train.shape}  │  Val : {Z_val.shape}"
        f"  │  Test : {Z_test.shape}  │  {time.time()-t0:.1f}s"
    )

    # ── Boucle principale few-shot ─────────────────────────────────────────
    resultats_jepa = []
    resultats_gnn  = []

    print(f"\n{'─'*70}")
    print(f"  Démarrage des expériences ({len(N_values)} valeurs de N × {n_runs} runs)")
    print(f"  Bio-JEPA : {args.epochs_probe} époques (sonde)  │  "
          f"GNN sup. : {args.epochs_supervised} époques")
    print(f"{'─'*70}")

    for N in N_values:
        print(f"\n  ── N = {N} ──────────────────────────────────────────────")
        N_eff = min(N, n_train)

        # --- Bio-JEPA few-shot ---
        t_jepa = time.time()
        res_jepa = experience_fewshot_biojep(
            Z_all_train=Z_all_train,
            y_all_train=y_all_train,
            Z_val=Z_val,
            y_val=y_val,
            Z_test=Z_test,
            y_test=y_test,
            N=N_eff,
            n_runs=n_runs,
            embedding_dim=cfg_m['embedding_dim'],
            hidden_dim=cfg_e['hidden_dim_sonde'],
            lr=cfg_e['lr_sonde'],
            num_epochs_sonde=args.epochs_probe,
            device=device,
            verbose=True,
        )
        resultats_jepa.append(res_jepa)
        mj = res_jepa['pearson_r']['mean'];  sj = res_jepa['pearson_r']['std']
        rj = res_jepa['spearman_r']['mean']; ej = res_jepa['rmse']['mean']
        print(
            f"    → Bio-JEPA  mean r={mj:+.4f}±{sj:.4f}"
            f"  ρ={rj:+.4f}  RMSE={ej:.4f}  ({time.time()-t_jepa:.1f}s)"
        )

        # --- GNN supervisé few-shot ---
        t_gnn = time.time()
        res_gnn = experience_fewshot_gnn(
            all_train_data=all_train_data,
            val_loader=val_loader,
            test_loader=test_loader,
            N=N_eff,
            n_runs=n_runs,
            cfg_m=cfg_m,
            cfg_t=cfg_t,
            num_epochs=args.epochs_supervised,
            device=device,
            verbose=True,
        )
        resultats_gnn.append(res_gnn)
        mg = res_gnn['pearson_r']['mean'];  sg = res_gnn['pearson_r']['std']
        rg = res_gnn['spearman_r']['mean']; eg = res_gnn['rmse']['mean']
        print(
            f"    → GNN sup.  mean r={mg:+.4f}±{sg:.4f}"
            f"  ρ={rg:+.4f}  RMSE={eg:.4f}  ({time.time()-t_gnn:.1f}s)"
        )

    # ── Résultats ──────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  Résultats finaux — {cible_label}")
    print(f"{'='*70}")

    afficher_courbe_ascii(resultats_jepa, resultats_gnn, N_values, n_runs, target_id)
    afficher_tableau(resultats_jepa, resultats_gnn, N_values, n_runs, target_id)

    # ── Sauvegarde JSON ────────────────────────────────────────────────────
    if args.save_json:
        os.makedirs(os.path.dirname(args.save_json) or '.', exist_ok=True)

        resultats_dict = {
            'target'    : target_id,
            'target_nom': nom_cible(target_id),
            'N_values'  : N_values,
            'n_runs'    : n_runs,
            'bio_jepa'  : [
                {
                    'N'          : N,
                    'pearson_r'  : res['pearson_r'],
                    'spearman_r' : res['spearman_r'],
                    'rmse'       : res['rmse'],
                }
                for N, res in zip(N_values, resultats_jepa)
            ],
            'gnn_supervise': [
                {
                    'N'          : N,
                    'pearson_r'  : res['pearson_r'],
                    'spearman_r' : res['spearman_r'],
                    'rmse'       : res['rmse'],
                }
                for N, res in zip(N_values, resultats_gnn)
            ],
            'config': {
                'checkpoint'        : checkpoint_path,
                'epochs_probe'      : args.epochs_probe,
                'epochs_supervised' : args.epochs_supervised,
                'seed'              : cfg_d['seed'],
                'target'            : target_id,
            },
        }
        with open(args.save_json, 'w', encoding='utf-8') as f:
            json.dump(resultats_dict, f, indent=2, ensure_ascii=False)
        print(f"  Résultats sauvegardés → {args.save_json}")

    # ── Conseil matplotlib ─────────────────────────────────────────────────
    json_path = args.save_json or f"results/few_shot_{target_id.lower()}.json"
    print(
        "  Pour une visualisation matplotlib :\n"
        "    import json, matplotlib.pyplot as plt\n"
        f"    d = json.load(open('{json_path}'))\n"
        "    N = d['N_values']\n"
        "    plt.errorbar(N, [x['pearson_r']['mean'] for x in d['bio_jepa']],\n"
        "                 yerr=[x['pearson_r']['std'] for x in d['bio_jepa']], label='Bio-JEPA')\n"
        "    plt.errorbar(N, [x['pearson_r']['mean'] for x in d['gnn_supervise']],\n"
        "                 yerr=[x['pearson_r']['std'] for x in d['gnn_supervise']], label='GNN')\n"
        "    plt.xscale('log'); plt.xlabel('N'); plt.ylabel('Pearson r'); plt.legend()\n"
    )


if __name__ == '__main__':
    main()
