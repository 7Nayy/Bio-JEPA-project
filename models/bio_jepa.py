"""
bio_jepa.py

Modèle Bio-JEPA complet : assemblage du Context Encoder, Target Encoder et Predictor.

Principe (adapté de I-JEPA pour les graphes moléculaires) :
──────────────────────────────────────────────────────────────
  1. MASQUAGE  : un sous-ensemble d'atomes (mask_ratio%) est masqué dans
                 le graphe → graphe_contexte (features des nœuds masqués = 0).

  2. ENCODAGE  :
     • context_encoder(graphe_contexte) → z_context  [avec gradient]
     • target_encoder(graphe_complet)   → z_target   [sans gradient, EMA]

  3. PRÉDICTION : predictor(z_context) → z_pred

  4. LOSS L2   : ||z_pred - z_target.detach()||²  dans l'espace latent.

Le target_encoder est une copie du context_encoder mise à jour par EMA
(voir training/ema.py). Cette asymétrie évite le mode d'effondrement
(collapsing) sans nécessiter de paires négatives (contrairement à SimCLR).

Références :
  - I-JEPA : Assran et al. (2023), CVPR
  - BYOL   : Grill et al. (2020), NeurIPS
  - BGRL   : Thakoor et al. (2022), adaptation graphe de BYOL
"""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch

from data.mol_graph import ATOM_FEATURE_DIM, BOND_FEATURE_DIM
from models.gnn_encoder import GNNEncoder
from models.predictor import Predictor


class BioJEPA(nn.Module):
    """
    Bio-JEPA : architecture JEPA pour la drug discovery.

    Args:
        node_dim             : Dimension des features d'atomes (défaut : 29).
        edge_dim             : Dimension des features de liaisons (défaut : 7).
        hidden_dim           : Dimension cachée du GNN (défaut : 128).
        embedding_dim        : Dimension de l'espace latent (défaut : 256).
        num_gnn_layers       : Nombre de couches GINEConv (défaut : 4).
        predictor_hidden_dim : Dimension cachée du Predictor (défaut : 512).
        predictor_num_layers : Couches cachées du Predictor (défaut : 2).
        mask_ratio           : Fraction d'atomes masqués comme contexte (défaut : 0.3).
        dropout              : Taux de dropout GNN et Predictor (défaut : 0.1).
    """

    def __init__(
        self,
        node_dim: int = ATOM_FEATURE_DIM,
        edge_dim: int = BOND_FEATURE_DIM,
        hidden_dim: int = 128,
        embedding_dim: int = 256,
        num_gnn_layers: int = 4,
        predictor_hidden_dim: int = 512,
        predictor_num_layers: int = 2,
        mask_ratio: float = 0.3,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.mask_ratio = mask_ratio
        self.embedding_dim = embedding_dim

        # Context Encoder — entraîné par gradient descent
        self.context_encoder = GNNEncoder(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            num_layers=num_gnn_layers,
            dropout=dropout,
        )

        # Target Encoder — copie profonde du context encoder, mis à jour par EMA
        # Aucun gradient ne passe par ce réseau (les paramètres ne sont pas
        # dans l'optimizer).
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for param in self.target_encoder.parameters():
            param.requires_grad = False

        # Predictor — mappe z_context → z̃_target
        self.predictor = Predictor(
            embedding_dim=embedding_dim,
            hidden_dim=predictor_hidden_dim,
            num_layers=predictor_num_layers,
            dropout=dropout,
        )

    # ------------------------------------------------------------------
    # Masquage du graphe
    # ------------------------------------------------------------------

    def _masquer_graphe(self, data: Batch) -> Batch:
        """
        Crée une version masquée du batch pour le Context Encoder.

        Stratégie : les features des nœuds sélectionnés aléatoirement
        (mask_ratio% du total) sont mises à zéro. La topologie du graphe
        (edge_index, edge_attr) est conservée.

        Note : le masquage est appliqué globalement sur le batch concaténé.
        Chaque graphe sera donc masqué de façon légèrement différente selon
        sa taille relative dans le batch.

        Args:
            data : Batch PyG (x, edge_index, edge_attr, batch).

        Returns:
            Batch masqué (même structure, x modifié).
        """
        num_nodes = data.x.size(0)
        n_masques = max(1, int(num_nodes * self.mask_ratio))

        # Sélection aléatoire des nœuds à masquer
        indices_masques = torch.randperm(
            num_nodes, device=data.x.device
        )[:n_masques]

        # Masque multiplicatif : 0 pour les nœuds masqués, 1 pour les autres
        masque = torch.ones(num_nodes, 1, device=data.x.device)
        masque[indices_masques] = 0.0

        # Application du masque (opération differentiable)
        x_masque = data.x * masque  # [N, node_dim]

        # Construction du batch masqué (clone pour éviter la modification en place)
        data_masque = data.clone()
        data_masque.x = x_masque

        return data_masque

    # ------------------------------------------------------------------
    # Passage avant (entraînement)
    # ------------------------------------------------------------------

    def forward(self, data: Batch) -> tuple:
        """
        Passage avant Bio-JEPA pour l'entraînement.

        Args:
            data : Batch PyG de graphes moléculaires.

        Returns:
            (loss, z_context, z_target, z_pred)
            - loss      : Scalaire — perte L2 dans l'espace latent.
            - z_context : Embeddings du Context Encoder [B, embedding_dim].
            - z_target  : Embeddings du Target Encoder [B, embedding_dim].
            - z_pred    : Embeddings prédits par le Predictor [B, embedding_dim].
        """
        # --- 1. Target Encoder (sans gradient) ---
        with torch.no_grad():
            z_target = self.target_encoder(
                data.x, data.edge_index, data.edge_attr, data.batch
            )
            # Normalisation L2 pour stabiliser l'espace latent
            z_target = F.normalize(z_target, dim=-1)

        # --- 2. Masquage pour le Context Encoder ---
        data_masque = self._masquer_graphe(data)

        # --- 3. Context Encoder (avec gradient) ---
        z_context = self.context_encoder(
            data_masque.x,
            data_masque.edge_index,
            data_masque.edge_attr,
            data_masque.batch,
        )

        # --- 4. Predictor : z_context → z̃_target ---
        z_pred = self.predictor(z_context)
        z_pred = F.normalize(z_pred, dim=-1)

        # --- 5. Loss L2 dans l'espace latent ---
        # z_target.detach() : aucun gradient ne traverse le target encoder
        loss = F.mse_loss(z_pred, z_target.detach())

        return loss, z_context, z_target, z_pred

    # ------------------------------------------------------------------
    # Encodage (inférence / évaluation)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode(self, data: Batch, encoder: str = 'target') -> torch.Tensor:
        """
        Encode un batch de graphes sans calculer de gradient.

        Args:
            data    : Batch PyG.
            encoder : 'target' (recommandé pour l'évaluation, plus stable)
                      ou 'context'.

        Returns:
            Embeddings [B, embedding_dim].
        """
        enc = (
            self.target_encoder if encoder == 'target'
            else self.context_encoder
        )
        self.eval()
        z = enc(data.x, data.edge_index, data.edge_attr, data.batch)
        return F.normalize(z, dim=-1)
