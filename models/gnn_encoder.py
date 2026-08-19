"""
gnn_encoder.py

Context Encoder et Target Encoder de Bio-JEPA.

Architecture GNN basée sur GIN (Graph Isomorphism Network) avec features
d'arêtes (GINEConv). Plus expressif que GCN pour capturer la structure
moléculaire (Xu et al., 2019 — "How Powerful Are Graph Neural Networks?").

L'encodeur transforme un graphe moléculaire en un vecteur d'embedding
de dimension fixe (embedding_dim) via :
  1. Projection des features de nœuds : node_dim → hidden_dim
  2. N couches GINEConv avec BatchNorm + ReLU + connexion résiduelle
  3. Readout global : concat(mean_pool, max_pool) → [2*hidden_dim]
  4. Projection finale : 2*hidden_dim → embedding_dim
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_mean_pool, global_max_pool

from data.mol_graph import ATOM_FEATURE_DIM, BOND_FEATURE_DIM


class GNNEncoder(nn.Module):
    """
    Encodeur GNN pour graphes moléculaires.

    Utilisé comme Context Encoder (entraîné par backpropagation)
    et comme Target Encoder (mis à jour par EMA).

    Args:
        node_dim      : Dimension des features de nœuds (défaut : ATOM_FEATURE_DIM=29).
        edge_dim      : Dimension des features d'arêtes (défaut : BOND_FEATURE_DIM=7).
        hidden_dim    : Dimension cachée des couches GNN (défaut : 128).
        embedding_dim : Dimension de l'embedding de sortie (défaut : 256).
        num_layers    : Nombre de couches GINEConv (défaut : 4).
        dropout       : Taux de dropout entre les couches (défaut : 0.1).
    """

    def __init__(
        self,
        node_dim: int = ATOM_FEATURE_DIM,
        edge_dim: int = BOND_FEATURE_DIM,
        hidden_dim: int = 128,
        embedding_dim: int = 256,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.num_layers = num_layers
        self.dropout = dropout

        # --- Projection initiale des features de nœuds ---
        # Transforme node_dim → hidden_dim pour uniformiser l'entrée des GINEConv
        self.proj_noeud = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )

        # --- Couches GINEConv ---
        # GINEConv gère la projection edge_dim → hidden_dim en interne
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for _ in range(num_layers):
            # MLP interne de la couche GIN
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, 2 * hidden_dim),
                nn.BatchNorm1d(2 * hidden_dim),
                nn.ReLU(),
                nn.Linear(2 * hidden_dim, hidden_dim),
            )
            # GINEConv : agrège les voisins en tenant compte des features d'arêtes
            self.convs.append(
                GINEConv(mlp, train_eps=True, edge_dim=edge_dim)
            )
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # --- Projection finale ---
        # Après readout concat(mean, max) : 2*hidden_dim → embedding_dim
        self.proj_finale = nn.Sequential(
            nn.Linear(2 * hidden_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """
        Passage avant : graphe moléculaire → embedding.

        Args:
            x          : Features des nœuds [N_total, node_dim].
            edge_index : Indices des arêtes [2, E_total].
            edge_attr  : Features des arêtes [E_total, edge_dim].
            batch      : Appartenance au graphe [N_total] (PyG batch vector).

        Returns:
            Embeddings des graphes [B, embedding_dim].
        """
        # Projection initiale des nœuds
        h = self.proj_noeud(x)  # [N, hidden_dim]

        # Couches GNN avec connexion résiduelle
        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            h_new = conv(h, edge_index, edge_attr)
            h_new = bn(h_new)
            h_new = F.relu(h_new)
            h_new = F.dropout(h_new, p=self.dropout, training=self.training)
            # Connexion résiduelle à partir de la 2e couche
            h = h + h_new if i > 0 else h_new

        # --- Readout global : agrège les nœuds par graphe ---
        z_mean = global_mean_pool(h, batch)           # [B, hidden_dim]
        z_max = global_max_pool(h, batch)             # [B, hidden_dim]
        z = torch.cat([z_mean, z_max], dim=-1)        # [B, 2*hidden_dim]

        # Projection vers l'espace latent
        z = self.proj_finale(z)                       # [B, embedding_dim]

        return z
