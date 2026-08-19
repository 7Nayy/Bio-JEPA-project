"""
predictor.py

Réseau de prédiction de Bio-JEPA (MLP léger).

Rôle : à partir de l'embedding produit par le Context Encoder (z_context),
prédire l'embedding que le Target Encoder aurait produit pour le graphe complet.

Contrairement au Target Encoder, le Predictor est mis à jour par backpropagation
à chaque step. Il reste intentionnellement léger pour forcer le Context Encoder
à apprendre de bonnes représentations, et non le Predictor à mémoriser.

Inspiré du design de I-JEPA (Assran et al., 2023) et de BYOL (Grill et al., 2020).
"""

import torch
import torch.nn as nn


class Predictor(nn.Module):
    """
    MLP prédicateur : z_context → z̃_target.

    Mappe l'embedding contexte (sous-graphe masqué) vers l'espace de
    l'embedding cible (graphe complet encodé par le Target Encoder).

    Architecture :
        Linear(embedding_dim → hidden_dim) → BN → GELU → Dropout
        [× num_layers]
        Linear(hidden_dim → embedding_dim)   # couche de sortie sans activation

    Args:
        embedding_dim : Dimension d'entrée et de sortie (espace latent commun).
        hidden_dim    : Dimension des couches cachées.
        num_layers    : Nombre de couches cachées.
        dropout       : Taux de dropout.
    """

    def __init__(
        self,
        embedding_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        couches = []
        dim_entree = embedding_dim

        # Couches cachées
        for _ in range(num_layers):
            couches += [
                nn.Linear(dim_entree, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),          # GELU : plus lisse que ReLU, courant en transformers
                nn.Dropout(dropout),
            ]
            dim_entree = hidden_dim

        # Couche de sortie : pas d'activation (espace latent non contraint)
        couches.append(nn.Linear(hidden_dim, embedding_dim))

        self.reseau = nn.Sequential(*couches)

    def forward(self, z_context: torch.Tensor) -> torch.Tensor:
        """
        Prédit l'embedding cible depuis l'embedding contexte.

        Args:
            z_context : Embeddings du Context Encoder [B, embedding_dim].

        Returns:
            z_pred    : Embeddings prédits [B, embedding_dim].
        """
        return self.reseau(z_context)
