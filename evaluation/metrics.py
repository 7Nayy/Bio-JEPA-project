"""
evaluation/metrics.py

Évaluation de la qualité des représentations apprises par Bio-JEPA.

Protocole standard en auto-supervisé :
  1. Geler les paramètres du Target Encoder (aucune mise à jour).
  2. Extraire les embeddings pour train / val / test.
  3. Entraîner une sonde MLP à 2 couches sur les embeddings figés.
  4. Évaluer la sonde sur le test set avec MSE, RMSE, MAE, Pearson et Spearman.

Pourquoi un MLP plutôt qu'une régression linéaire ?
  La relation entre embedding et pChEMBL peut être non-linéaire même si les
  représentations sont bonnes. Le MLP 2 couches capte ces non-linéarités légères
  tout en restant peu expressif (pas de sur-apprentissage de la sonde elle-même).
  Avec Pearson r = 0.06 en linéaire, les directions utiles existent probablement
  mais ne sont pas linéairement séparables → le MLP doit mieux les exploiter.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats


# ---------------------------------------------------------------------------
# Sonde MLP (modèle d'évaluation)
# ---------------------------------------------------------------------------

class SondeMLP(nn.Module):
    """
    Sonde MLP à 2 couches cachées sur les embeddings figés.

    Plus flexible qu'une régression linéaire pour capturer des relations
    non-linéaires dans l'espace latent tout en restant légère.

    Architecture :
        Linear(D → H) → BN → ReLU
        Linear(H → H) → BN → ReLU
        Linear(H → 1)

    Args:
        embedding_dim : Dimension D des embeddings d'entrée.
        hidden_dim    : Dimension H des couches cachées (défaut : 256).
    """

    def __init__(self, embedding_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.reseau = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z : Embeddings [B, embedding_dim].

        Returns:
            Prédictions d'affinité [B].
        """
        return self.reseau(z).squeeze(-1)


# ---------------------------------------------------------------------------
# Extraction des embeddings
# ---------------------------------------------------------------------------

def extraire_embeddings(model, loader, device, encoder: str = 'target'):
    """
    Extrait les embeddings et labels de tout un DataLoader.

    Args:
        model   : Modèle BioJEPA (en mode eval).
        loader  : DataLoader PyG.
        device  : Dispositif de calcul.
        encoder : 'target' (recommandé) ou 'context'.

    Returns:
        (embeddings, labels) — tableaux numpy de formes [N, D] et [N].
    """
    model.eval()
    embeddings_liste, labels_liste = [], []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            z = model.encode(batch, encoder=encoder)
            embeddings_liste.append(z.cpu().numpy())
            labels_liste.append(batch.y.cpu().numpy().flatten())

    return np.vstack(embeddings_liste), np.concatenate(labels_liste)


# ---------------------------------------------------------------------------
# Calcul des métriques
# ---------------------------------------------------------------------------

def calculer_metriques(y_vrai: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Calcule un ensemble complet de métriques de régression.

    Args:
        y_vrai : Valeurs vraies (pChEMBL).
        y_pred : Valeurs prédites.

    Returns:
        Dictionnaire :
          - mse, rmse, mae        : erreurs quadratiques/absolues
          - pearson_r, pearson_p  : corrélation de Pearson et p-valeur
          - spearman_r, spearman_p: corrélation de Spearman et p-valeur
    """
    residus = y_vrai - y_pred
    mse = float(np.mean(residus ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(residus)))

    pearson_r, pearson_p = stats.pearsonr(y_vrai, y_pred)
    spearman_r, spearman_p = stats.spearmanr(y_vrai, y_pred)

    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'pearson_r': float(pearson_r),
        'pearson_p': float(pearson_p),
        'spearman_r': float(spearman_r),
        'spearman_p': float(spearman_p),
    }


# ---------------------------------------------------------------------------
# Évaluation complète par sonde MLP
# ---------------------------------------------------------------------------

def evaluer_sonde_mlp(
    model,
    train_loader,
    val_loader,
    test_loader,
    device,
    embedding_dim: int = 256,
    hidden_dim: int = 256,
    lr: float = 1e-3,
    num_epochs: int = 200,
    verbose: bool = True,
) -> dict:
    """
    Évalue la qualité des représentations Bio-JEPA par sonde MLP 2 couches.

    Pipeline :
      1. Extraction des embeddings figés (Target Encoder, aucune MAJ du modèle).
      2. Entraînement d'un MLP léger sur les embeddings train.
      3. Sélection du meilleur état sur val (early stopping par val_MSE).
      4. Calcul des métriques finales sur test.

    Args:
        model         : BioJEPA pré-entraîné.
        train_loader  : DataLoader d'entraînement.
        val_loader    : DataLoader de validation.
        test_loader   : DataLoader de test.
        device        : Dispositif de calcul.
        embedding_dim : Dimension des embeddings (défaut : 256).
        hidden_dim    : Dimension cachée de la sonde MLP (défaut : 256).
        lr            : LR de la sonde (défaut : 1e-3).
        num_epochs    : Époques d'entraînement de la sonde (défaut : 200).
        verbose       : Affiche les résultats.

    Returns:
        Dictionnaire de métriques sur le test set.
    """
    if verbose:
        print("\n[Évaluation] Extraction des embeddings (Target Encoder figé)...")

    # 1. Extraction (une seule passe, pas de gradient)
    z_train, y_train = extraire_embeddings(model, train_loader, device)
    z_val, y_val = extraire_embeddings(model, val_loader, device)
    z_test, y_test = extraire_embeddings(model, test_loader, device)

    if verbose:
        print(
            f"  Train : {z_train.shape[0]} | "
            f"Val : {z_val.shape[0]} | "
            f"Test : {z_test.shape[0]}"
        )
        print(f"  Sonde MLP : {embedding_dim} → {hidden_dim} → {hidden_dim} → 1")

    # 2. Conversion en tenseurs
    def _t(arr):
        return torch.tensor(arr, dtype=torch.float, device=device)

    z_train_t, y_train_t = _t(z_train), _t(y_train)
    z_val_t, y_val_t = _t(z_val), _t(y_val)
    z_test_t = _t(z_test)

    # 3. Sonde MLP
    sonde = SondeMLP(embedding_dim, hidden_dim).to(device)
    optimizer_sonde = torch.optim.Adam(sonde.parameters(), lr=lr, weight_decay=1e-4)
    scheduler_sonde = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_sonde, T_max=num_epochs, eta_min=lr * 0.01
    )

    meilleure_val_mse = float('inf')
    meilleures_params = None

    for _ in range(num_epochs):
        sonde.train()
        optimizer_sonde.zero_grad()
        loss_train = F.mse_loss(sonde(z_train_t), y_train_t)
        loss_train.backward()
        optimizer_sonde.step()
        scheduler_sonde.step()

        sonde.eval()
        with torch.no_grad():
            val_mse = F.mse_loss(sonde(z_val_t), y_val_t).item()

        if val_mse < meilleure_val_mse:
            meilleure_val_mse = val_mse
            meilleures_params = {k: v.clone() for k, v in sonde.state_dict().items()}

    # 4. Évaluation finale sur le test set
    sonde.load_state_dict(meilleures_params)
    sonde.eval()
    with torch.no_grad():
        pred_test = sonde(z_test_t).cpu().numpy()

    metriques = calculer_metriques(y_test, pred_test)

    if verbose:
        print("\n[Évaluation] Résultats de la sonde MLP sur le test set :")
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

    return metriques
