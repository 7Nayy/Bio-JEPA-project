"""
baseline.py

Baseline GNN supervisée pour comparer avec Bio-JEPA sur ChEMBL251.

Protocole de comparaison équitable (fair comparison) :
────────────────────────────────────────────────────────
  Même encodeur GNN   : GINEConv × 4, hidden=128, embedding=256
  Mêmes splits        : random_split(seed=42, 80/10/10)
  Même optimizer      : AdamW(lr=1e-4, wd=1e-5)
  Même scheduler      : CosineAnnealingLR
  Même gradient clip  : 1.0
  Mêmes métriques     : MSE, RMSE, MAE, Pearson r, Spearman ρ
  Mêmes époques       : 300

Différence fondamentale avec Bio-JEPA :
  • Baseline   : le GNN voit les labels pChEMBL à chaque step d'entraînement
                 (supervision directe, end-to-end).
  • Bio-JEPA   : le GNN ne voit JAMAIS les labels pendant les 300 époques
                 (auto-supervisé JEPA). Les labels n'interviennent que pour
                 la sonde MLP finale (200 époques sur embeddings figés).

Usage :
    python baseline.py                     # ChEMBL251, téléchargement auto
    python baseline.py --csv data.csv      # CSV local
    python baseline.py --demo              # Dataset synthétique (test rapide)
    python baseline.py --epochs 100        # Override du nombre d'époques
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from scipy import stats
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.loader import DataLoader

from data.chembl_dataset import ChEMBLDataset, creer_dataloaders, creer_dataset_demo
from data.mol_graph import ATOM_FEATURE_DIM, BOND_FEATURE_DIM
from evaluation.metrics import calculer_metriques
from models.gnn_encoder import GNNEncoder


# ---------------------------------------------------------------------------
# Modèle supervisé : GNNEncoder + Tête de régression MLP
# ---------------------------------------------------------------------------

class GNNRegresseur(nn.Module):
    """
    GNN supervisé end-to-end pour la prédiction d'affinité pChEMBL.

    L'encodeur est strictement identique au Context Encoder de Bio-JEPA
    (mêmes hyperparamètres, mêmes couches GINEConv). La seule différence :
    une tête de régression MLP est branchée directement en sortie.

    Tête de régression : embedding_dim → mlp_hidden → mlp_hidden → 1
    (même profondeur que la SondeMLP de l'évaluation Bio-JEPA)

    Args:
        node_dim      : Dimension des features d'atomes (défaut : 29).
        edge_dim      : Dimension des features de liaisons (défaut : 7).
        hidden_dim    : Dimension cachée du GNN (défaut : 128).
        embedding_dim : Dimension de l'embedding intermédiaire (défaut : 256).
        num_layers    : Nombre de couches GINEConv (défaut : 4).
        mlp_hidden    : Dimension des couches de la tête MLP (défaut : 256).
        dropout       : Dropout dans le GNN et la tête (défaut : 0.1).
    """

    def __init__(
        self,
        node_dim: int = ATOM_FEATURE_DIM,
        edge_dim: int = BOND_FEATURE_DIM,
        hidden_dim: int = 128,
        embedding_dim: int = 256,
        num_layers: int = 4,
        mlp_hidden: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Encodeur GNN — importé depuis models/gnn_encoder.py, identique à Bio-JEPA
        self.encodeur = GNNEncoder(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

        # Tête de régression MLP (2 couches cachées avec BN + ReLU + Dropout)
        # Même profondeur que SondeMLP dans evaluation/metrics.py
        self.tete = nn.Sequential(
            nn.Linear(embedding_dim, mlp_hidden),
            nn.BatchNorm1d(mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.BatchNorm1d(mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1),
        )

    def forward(self, data) -> torch.Tensor:
        """
        Passage avant complet : graphe → prédiction d'affinité.

        Args:
            data : Batch PyG (x, edge_index, edge_attr, batch).

        Returns:
            Prédictions pChEMBL [B].
        """
        z = self.encodeur(data.x, data.edge_index, data.edge_attr, data.batch)
        return self.tete(z).squeeze(-1)

    @torch.no_grad()
    def encoder_uniquement(self, data) -> torch.Tensor:
        """
        Extrait l'embedding sans la tête de régression.
        Utile pour visualiser l'espace latent appris en supervisé.
        """
        self.eval()
        return self.encodeur(data.x, data.edge_index, data.edge_attr, data.batch)


# ---------------------------------------------------------------------------
# Utilitaire sparkline ASCII (identique au Trainer Bio-JEPA)
# ---------------------------------------------------------------------------

def _sparkline(valeurs: list, largeur: int = 60) -> str:
    """Génère une sparkline ASCII depuis une liste de valeurs."""
    BLOCS = ' ▁▂▃▄▅▆▇█'
    vals_valides = [v for v in valeurs if v == v]
    if not vals_valides:
        return '(aucune valeur)'
    mini, maxi = min(vals_valides), max(vals_valides)
    plage = maxi - mini
    if len(valeurs) > largeur:
        pas = len(valeurs) / largeur
        valeurs = [valeurs[int(i * pas)] for i in range(largeur)]

    def _bloc(v):
        if v != v:
            return ' '
        if plage < 1e-8:
            return BLOCS[4]
        return BLOCS[min(int((v - mini) / plage * 8), 8)]

    return ''.join(_bloc(v) for v in valeurs)


# ---------------------------------------------------------------------------
# Boucle d'entraînement supervisée
# ---------------------------------------------------------------------------

def entrainer_baseline(
    model: GNNRegresseur,
    train_loader,
    val_loader,
    num_epochs: int = 300,
    lr: float = 1e-4,
    weight_decay: float = 1e-5,
    gradient_clip: float = 1.0,
    checkpoint_dir: str = 'checkpoints',
    device=None,
    verbose: bool = True,
) -> tuple:
    """
    Entraîne le GNNRegresseur en supervision directe (MSE sur pChEMBL).

    Identique à BioJEPATrainer pour tous les paramètres d'optimisation.
    Le Pearson r sur val est calculé à chaque époque directement depuis
    les prédictions du modèle (sans sonde auxiliaire).

    Args:
        model         : GNNRegresseur initialisé.
        train_loader  : DataLoader PyG d'entraînement.
        val_loader    : DataLoader PyG de validation.
        num_epochs    : Nombre d'époques (défaut : 300).
        lr            : Learning rate initial AdamW (défaut : 1e-4).
        weight_decay  : Pénalité L2 (défaut : 1e-5).
        gradient_clip : Norme max du gradient (défaut : 1.0).
        checkpoint_dir: Dossier de sauvegarde.
        device        : Dispositif de calcul. Auto-détecté si None.
        verbose       : Affiche les métriques toutes les 5 époques.

    Returns:
        (historique, model, device)
        - historique : dict avec train_loss, val_loss, pearson_r, lr par époque.
        - model      : GNNRegresseur avec les meilleurs poids (best val_loss).
        - device     : Dispositif utilisé.
    """
    # Auto-détection du dispositif (identique au Trainer Bio-JEPA)
    if device is None:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    elif isinstance(device, str):
        device = torch.device(device)

    model = model.to(device)
    os.makedirs(checkpoint_dir, exist_ok=True)
    chemin_best = os.path.join(checkpoint_dir, 'baseline_best.pt')

    # Optimiseur et scheduler — mêmes paramètres que Bio-JEPA
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = num_epochs * len(train_loader)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=lr * 0.01)

    historique = {
        'train_loss': [],
        'val_loss': [],
        'pearson_r': [],   # Pearson r val calculé depuis les prédictions directes
        'lr': [],
    }
    meilleure_val_loss = float('inf')

    print(f"{'─'*70}")
    print(f"  Baseline GNN supervisée — {device}")
    print(f"  Époques : {num_epochs}  │  "
          f"Train : {len(train_loader.dataset):,}  │  "
          f"Val : {len(val_loader.dataset):,}")
    print(f"{'─'*70}")

    for epoch in range(1, num_epochs + 1):
        # ── Entraînement ──────────────────────────────────────────────────
        model.train()
        debut = time.time()
        losses_train = []

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            pred = model(batch)
            # Cible : pChEMBL (scalaire par molécule)
            y = batch.y.view(-1)
            loss = F.mse_loss(pred, y)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=gradient_clip
            )
            optimizer.step()
            scheduler.step()
            losses_train.append(loss.item())

        train_loss = sum(losses_train) / len(losses_train)

        # ── Validation + Pearson r ─────────────────────────────────────────
        model.eval()
        losses_val, y_vrai_list, y_pred_list = [], [], []

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                pred = model(batch)
                y = batch.y.view(-1)
                losses_val.append(F.mse_loss(pred, y).item())
                y_vrai_list.append(y.cpu().numpy())
                y_pred_list.append(pred.cpu().numpy())

        val_loss = sum(losses_val) / len(losses_val)
        y_vrai = np.concatenate(y_vrai_list)
        y_pred_val = np.concatenate(y_pred_list)
        pearson_r, _ = stats.pearsonr(y_vrai, y_pred_val)
        lr_actuel = scheduler.get_last_lr()[0]

        # ── Historique ────────────────────────────────────────────────────
        historique['train_loss'].append(train_loss)
        historique['val_loss'].append(val_loss)
        historique['pearson_r'].append(float(pearson_r))
        historique['lr'].append(lr_actuel)

        # ── Checkpoint du meilleur modèle (critère : val MSE) ─────────────
        if val_loss < meilleure_val_loss:
            meilleure_val_loss = val_loss
            torch.save(model.state_dict(), chemin_best)

        # ── Affichage ─────────────────────────────────────────────────────
        if verbose and (epoch % 5 == 0 or epoch == 1 or epoch == num_epochs):
            duree = time.time() - debut
            print(
                f"  Ep {epoch:4d}/{num_epochs}"
                f"  │ train {train_loss:.4f}"
                f"  │ val {val_loss:.4f}"
                f"  │ r {pearson_r:+.3f}"
                f"  │ lr {lr_actuel:.1e}"
                f"  │ {duree:.1f}s"
            )

    # ── Résumé + sparkline ─────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print(f"  Entraînement terminé  │  Meilleure val_loss : {meilleure_val_loss:.4f}")

    rs = historique['pearson_r']
    r_min, r_max, r_final = min(rs), max(rs), rs[-1]
    sparkline = _sparkline(rs)
    print(f"\n  Pearson r par époque (val set, prédictions directes) :")
    print(f"  Ep 1 {'─'*3} Ep {num_epochs}")
    print(f"  {sparkline}")
    print(f"  min {r_min:+.3f}  │  max {r_max:+.3f}  │  final {r_final:+.3f}")
    print(f"{'─'*70}\n")

    # Rechargement des meilleurs poids pour l'évaluation
    model.load_state_dict(
        torch.load(chemin_best, map_location=device, weights_only=True)
    )
    return historique, model, device


# ---------------------------------------------------------------------------
# Évaluation sur le test set
# ---------------------------------------------------------------------------

def evaluer_baseline(model: GNNRegresseur, test_loader, device) -> dict:
    """
    Évalue le GNNRegresseur sur le test set.

    Args:
        model       : GNNRegresseur (meilleurs poids chargés).
        test_loader : DataLoader PyG de test.
        device      : Dispositif de calcul.

    Returns:
        Dictionnaire de métriques (MSE, RMSE, MAE, Pearson r, Spearman ρ).
    """
    model.eval()
    y_vrai_list, y_pred_list = [], []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            pred = model(batch)
            y_vrai_list.append(batch.y.view(-1).cpu().numpy())
            y_pred_list.append(pred.cpu().numpy())

    y_vrai = np.concatenate(y_vrai_list)
    y_pred = np.concatenate(y_pred_list)
    return calculer_metriques(y_vrai, y_pred)


# ---------------------------------------------------------------------------
# Tableau comparatif
# ---------------------------------------------------------------------------

# Résultats Bio-JEPA obtenus sur CHEMBL251 (6739 mol., 300 époques)
# Source : run du 2026-03-01, sonde MLP 2 couches sur Target Encoder figé
RESULTATS_BIOJEP = {
    'pearson_r': 0.787,
    'spearman_r': 0.794,
    'rmse': 0.807,
    'mae': 0.604,
}

# Affichage pour ces métriques (ordre + libellés)
METRIQUES_COMPARAISON = [
    ('pearson_r',  'Pearson r',  True),   # (clé, libellé, plus_grand_est_mieux)
    ('spearman_r', 'Spearman ρ', True),
    ('rmse',       'RMSE',       False),
    ('mae',        'MAE',        False),
]


def afficher_comparaison(metriques_baseline: dict):
    """
    Affiche un tableau comparatif Baseline vs Bio-JEPA.

    Le ✓ indique le meilleur des deux modèles pour chaque métrique.

    Args:
        metriques_baseline : Résultats de evaluer_baseline() sur le test set.
    """
    print("=" * 70)
    print("  Comparaison — Test set ChEMBL251 (6 739 molécules, seed=42)")
    print("=" * 70)
    print(f"  {'Métrique':<14}  {'GNN supervisé':>16}  {'Bio-JEPA':>16}  Δ (base−JEPA)")
    print(f"  {'─'*14}  {'─'*16}  {'─'*16}  {'─'*14}")

    for cle, libelle, grand_mieux in METRIQUES_COMPARAISON:
        v_base = metriques_baseline.get(cle)
        v_jepa = RESULTATS_BIOJEP.get(cle)

        if v_base is None or v_jepa is None:
            continue

        delta = v_base - v_jepa
        signe = '+' if delta > 0 else ''

        # Déterminer le gagnant
        if grand_mieux:
            gagnant_base = v_base > v_jepa
        else:
            gagnant_base = v_base < v_jepa

        marqueur_base = ' ✓' if gagnant_base else '  '
        marqueur_jepa = ' ✓' if not gagnant_base else '  '

        print(
            f"  {libelle:<14}  {v_base:>15.4f}{marqueur_base}"
            f"  {v_jepa:>15.4f}{marqueur_jepa}"
            f"  {signe}{delta:.4f}"
        )

    print("=" * 70)
    print(
        "\n  Interprétation :\n"
        "  • GNN supervisé  : accès aux labels pChEMBL à chaque step (300 × N_batch fois).\n"
        "  • Bio-JEPA        : zéro label pendant le pré-entraînement JEPA.\n"
        "                      Labels vus uniquement pour la sonde MLP finale\n"
        "                      (200 époques sur embeddings figés).\n"
        "  Si Bio-JEPA > GNN supervisé sur Pearson r → les représentations\n"
        "  auto-supervisées transfèrent mieux que la supervision directe.\n"
    )


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------

def charger_config(chemin: str) -> dict:
    with open(chemin) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description='Baseline GNN supervisée — comparaison avec Bio-JEPA',
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        '--config', default='configs/default.yaml',
        help='Fichier de configuration (défaut : configs/default.yaml)'
    )
    parser.add_argument(
        '--demo', action='store_true',
        help='Dataset synthétique (test rapide du pipeline)'
    )
    parser.add_argument(
        '--csv', type=str, default=None,
        help="CSV local avec colonnes 'smiles' et 'pchembl_value'"
    )
    parser.add_argument(
        '--target', type=str, default=None,
        help="ID de cible ChEMBL (défaut : CHEMBL251)"
    )
    parser.add_argument(
        '--epochs', type=int, default=None,
        help='Surcharge le nombre d\'époques de la config'
    )
    args = parser.parse_args()

    config = charger_config(args.config)
    if args.epochs:
        config['entrainement']['num_epochs'] = args.epochs

    cfg_m = config['modele']
    cfg_t = config['entrainement']
    cfg_d = config['donnees']

    print("=" * 70)
    print("  Baseline — GNN Supervisé (référence pour Bio-JEPA)")
    print("  Cible : ChEMBL251 (Récepteur Adénosine A2A)")
    print("=" * 70)

    # ── Données — splits identiques à train.py (même seed, même ratio) ────
    if args.demo:
        print("\n[Données] Mode démonstration (dataset synthétique)")
        data_list = creer_dataset_demo(n_molecules=500)
        n = len(data_list)
        n_train = int(n * cfg_d['train_ratio'])
        n_val = int(n * cfg_d['val_ratio'])

        # Seed identique à train.py → même partition train/val/test
        torch.manual_seed(cfg_d['seed'])
        indices = torch.randperm(n).tolist()

        bs = cfg_t['batch_size']
        train_loader = DataLoader(
            [data_list[i] for i in indices[:n_train]], batch_size=bs, shuffle=True
        )
        val_loader = DataLoader(
            [data_list[i] for i in indices[n_train:n_train + n_val]], batch_size=bs
        )
        test_loader = DataLoader(
            [data_list[i] for i in indices[n_train + n_val:]], batch_size=bs
        )
    else:
        print("\n[Données] Chargement du dataset ChEMBL...")
        # Même appel que dans train.py → même cache PyG → même dataset
        dataset = ChEMBLDataset(
            root=cfg_d['root'],
            csv_path=args.csv,
            target_chembl_id=args.target or 'CHEMBL251',
        )
        # creer_dataloaders(seed=42) → splits identiques à ceux de Bio-JEPA
        train_loader, val_loader, test_loader = creer_dataloaders(
            dataset,
            train_ratio=cfg_d['train_ratio'],
            val_ratio=cfg_d['val_ratio'],
            batch_size=cfg_t['batch_size'],
            seed=cfg_d['seed'],   # ← même graine = mêmes molécules dans chaque split
        )

    print(
        f"\n  Train : {len(train_loader.dataset):,} mol."
        f"  │  Val : {len(val_loader.dataset):,} mol."
        f"  │  Test : {len(test_loader.dataset):,} mol."
    )

    # ── Modèle ────────────────────────────────────────────────────────────
    print("\n[Modèle] GNNRegresseur :")
    print(f"  GINEConv × {cfg_m['num_gnn_layers']}"
          f"  │  hidden={cfg_m['hidden_dim']}"
          f"  │  embedding={cfg_m['embedding_dim']}"
          f"  │  tête MLP {cfg_m['embedding_dim']}→{cfg_m['embedding_dim']}→1")

    model = GNNRegresseur(
        hidden_dim=cfg_m['hidden_dim'],
        embedding_dim=cfg_m['embedding_dim'],
        num_layers=cfg_m['num_gnn_layers'],
        mlp_hidden=cfg_m['embedding_dim'],
        dropout=cfg_m['dropout'],
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Paramètres entraînables : {n_params:,}")

    # ── Entraînement ──────────────────────────────────────────────────────
    print(f"\n[Entraînement] Démarrage ({cfg_t['num_epochs']} époques)...")
    historique, model, device = entrainer_baseline(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=cfg_t['num_epochs'],
        lr=cfg_t['lr'],
        weight_decay=cfg_t['weight_decay'],
        gradient_clip=cfg_t['gradient_clip'],
        checkpoint_dir=config['checkpoints']['dir'],
        verbose=True,
    )

    # ── Évaluation sur le test set ─────────────────────────────────────────
    print("[Évaluation] Test set...")
    metriques = evaluer_baseline(model, test_loader, device)

    print("\n[Évaluation] Métriques complètes :")
    print(f"  MSE          : {metriques['mse']:.4f}")
    print(f"  RMSE         : {metriques['rmse']:.4f}")
    print(f"  MAE          : {metriques['mae']:.4f}")
    print(
        f"  Pearson  r   : {metriques['pearson_r']:.4f}"
        f"  (p = {metriques['pearson_p']:.2e})"
    )
    print(
        f"  Spearman ρ   : {metriques['spearman_r']:.4f}"
        f"  (p = {metriques['spearman_p']:.2e})"
    )

    # ── Tableau comparatif ────────────────────────────────────────────────
    print()
    afficher_comparaison(metriques)


if __name__ == '__main__':
    main()
