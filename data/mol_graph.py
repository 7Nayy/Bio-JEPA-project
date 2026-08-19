"""
mol_graph.py

Conversion de chaînes SMILES en graphes moléculaires PyTorch Geometric.
Chaque atome devient un nœud, chaque liaison une arête (bidirectionnelle).
"""

import torch
from rdkit import Chem
from torch_geometric.data import Data


# ---------------------------------------------------------------------------
# Vocabulaire des features atomiques
# ---------------------------------------------------------------------------

# Symboles chimiques les plus fréquents dans les médicaments
SYMBOLES_ATOMES = ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I', 'P', 'Si']

# Charges formelles courantes
CHARGES_FORMELLES = [-2, -1, 0, 1, 2]

# Types d'hybridisation
HYBRIDISATIONS = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]

# Types de liaisons chimiques
TYPES_LIAISONS = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]

# ---------------------------------------------------------------------------
# Dimensions des vecteurs de features (utiles pour configurer les modèles)
# ---------------------------------------------------------------------------
# one_hot retourne len(liste) + 1 valeurs (la dernière = catégorie inconnue)
ATOM_FEATURE_DIM = (
    (len(SYMBOLES_ATOMES) + 1)    # type atome     → 11
    + (len(CHARGES_FORMELLES) + 1)  # charge formelle → 6
    + (len(HYBRIDISATIONS) + 1)     # hybridisation   → 6
    + 1                             # aromaticité     → 1
    + 5                             # nb hydrogènes   → 5
)  # Total : 29

BOND_FEATURE_DIM = (
    (len(TYPES_LIAISONS) + 1)  # type liaison → 5
    + 1                        # conjuguée    → 1
    + 1                        # en cycle     → 1
)  # Total : 7


# ---------------------------------------------------------------------------
# Fonctions utilitaires
# ---------------------------------------------------------------------------

def _one_hot(valeur, categories: list) -> list:
    """
    Encodage one-hot avec une catégorie 'autre' en dernière position.

    Args:
        valeur   : Valeur à encoder.
        categories : Liste des catégories connues.

    Returns:
        Vecteur de longueur len(categories) + 1.
    """
    vecteur = [0] * (len(categories) + 1)
    if valeur in categories:
        vecteur[categories.index(valeur)] = 1
    else:
        vecteur[-1] = 1  # catégorie inconnue
    return vecteur


def _features_atome(atom) -> list:
    """
    Calcule le vecteur de features d'un atome RDKit.

    Dimensions : 11 + 6 + 6 + 1 + 5 = 29
    """
    feat = []
    feat += _one_hot(atom.GetSymbol(), SYMBOLES_ATOMES)           # 11
    feat += _one_hot(atom.GetFormalCharge(), CHARGES_FORMELLES)   # 6
    feat += _one_hot(atom.GetHybridization(), HYBRIDISATIONS)     # 6
    feat += [int(atom.GetIsAromatic())]                           # 1
    # Nombre d'hydrogènes : plafonné à 4
    n_h = min(atom.GetTotalNumHs(), 4)
    feat += [int(n_h == i) for i in range(5)]                     # 5
    return feat  # Total : 29


def _features_liaison(bond) -> list:
    """
    Calcule le vecteur de features d'une liaison RDKit.

    Dimensions : 5 + 1 + 1 = 7
    """
    feat = []
    feat += _one_hot(bond.GetBondType(), TYPES_LIAISONS)  # 5
    feat += [int(bond.GetIsConjugated())]                  # 1
    feat += [int(bond.IsInRing())]                         # 1
    return feat  # Total : 7


# ---------------------------------------------------------------------------
# Conversion principale
# ---------------------------------------------------------------------------

def smiles_vers_graphe(smiles: str) -> Data | None:
    """
    Convertit un SMILES en objet torch_geometric.data.Data.

    Le graphe est non dirigé : chaque liaison génère deux arêtes (i→j et j→i).

    Args:
        smiles : Chaîne SMILES de la molécule.

    Returns:
        Objet Data(x, edge_index, edge_attr) ou None si le SMILES est invalide.
        - x          : [N, ATOM_FEATURE_DIM]  features des atomes
        - edge_index : [2, 2*E]               indices des arêtes (bidirectionnel)
        - edge_attr  : [2*E, BOND_FEATURE_DIM] features des liaisons
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None  # SMILES invalide → ignoré

    # --- Nœuds (atomes) ---
    x = torch.tensor(
        [_features_atome(a) for a in mol.GetAtoms()],
        dtype=torch.float
    )  # [N, 29]

    # --- Arêtes (liaisons) ---
    src, dst, edge_attrs = [], [], []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        feat = _features_liaison(bond)
        # Graphe non dirigé : arête dans les deux sens
        src += [i, j]
        dst += [j, i]
        edge_attrs += [feat, feat]

    if len(src) == 0:
        # Atome isolé sans liaisons
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, BOND_FEATURE_DIM), dtype=torch.float)
    else:
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
