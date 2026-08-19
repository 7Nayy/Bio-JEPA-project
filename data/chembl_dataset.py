"""
chembl_dataset.py

Dataset PyTorch Geometric pour les molécules ChEMBL avec affinités de liaison.

Deux modes d'utilisation :
  1. Chargement depuis un CSV local (colonnes 'smiles' et 'pchembl_value').
  2. Téléchargement automatique via l'API ChEMBL Web Services.

La valeur pChEMBL est un score standardisé d'affinité = -log10(Ki/IC50/Kd en molaire).
Plus il est élevé, plus la molécule est active (typiquement entre 4 et 10).
"""

import os
import shutil

import torch
import pandas as pd
from torch.utils.data import random_split
from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import DataLoader

from data.mol_graph import smiles_vers_graphe


# ---------------------------------------------------------------------------
# Dataset principal
# ---------------------------------------------------------------------------

class ChEMBLDataset(InMemoryDataset):
    """
    Dataset de molécules ChEMBL avec affinités pChEMBL.

    Hérite de InMemoryDataset : les graphes sont pré-calculés une fois
    et mis en cache sur disque pour accélérer les epochs suivants.

    Args:
        root             : Répertoire racine (cache des données traitées).
        csv_path         : Chemin vers un CSV avec colonnes 'smiles' et
                           'pchembl_value'. Prioritaire sur target_chembl_id.
        target_chembl_id : Identifiant ChEMBL de la protéine cible
                           (ex: 'CHEMBL251' = récepteur A2A).
                           Utilisé si csv_path est None.
        transform        : Transformation PyG appliquée à chaque graphe (lazy).
        pre_transform    : Transformation PyG appliquée avant la mise en cache.
    """

    def __init__(
        self,
        root: str,
        csv_path: str = None,
        target_chembl_id: str = None,
        transform=None,
        pre_transform=None,
    ):
        self.csv_path = csv_path
        self.target_chembl_id = target_chembl_id
        super().__init__(root, transform, pre_transform)
        # Chargement du dataset pré-traité
        self.data, self.slices = torch.load(self.processed_paths[0],
                                            weights_only=False)

    @property
    def raw_file_names(self):
        """Nom du fichier brut attendu dans raw_dir."""
        if self.csv_path:
            return [os.path.basename(self.csv_path)]
        return ['chembl_data.csv']

    @property
    def processed_file_names(self):
        """Nom du fichier traité (cache des graphes PyG)."""
        return ['data.pt']

    def download(self):
        """
        Télécharge ou copie les données brutes vers raw_dir.
        Appelé automatiquement si le fichier brut est absent.
        """
        if self.csv_path and os.path.exists(self.csv_path):
            # Copie du CSV local
            shutil.copy(self.csv_path, self.raw_paths[0])
        elif self.target_chembl_id:
            self._telecharger_depuis_api()
        else:
            raise ValueError(
                "Fournissez csv_path OU target_chembl_id.\n"
                "Exemple : ChEMBLDataset(root='data/chembl', csv_path='molecules.csv')\n"
                "CSV attendu : colonnes 'smiles' et 'pchembl_value'."
            )

    def _telecharger_depuis_api(self):
        """
        Télécharge les données d'activité depuis l'API ChEMBL.

        Nécessite : pip install chembl-webresource-client
        """
        try:
            from chembl_webresource_client.new_client import new_client
        except ImportError:
            raise ImportError(
                "Installez le client ChEMBL : pip install chembl-webresource-client"
            )

        print(f"[ChEMBL] Téléchargement pour la cible {self.target_chembl_id}...")
        activite = new_client.activity

        # Requête : Ki, IC50 ou Kd avec une valeur pChEMBL disponible
        resultats = activite.filter(
            target_chembl_id=self.target_chembl_id,
            standard_type__in=['Ki', 'IC50', 'Kd'],
            pchembl_value__isnull=False,
        ).only(['canonical_smiles', 'pchembl_value', 'standard_type'])

        df = pd.DataFrame(list(resultats))
        df = df.rename(columns={'canonical_smiles': 'smiles'})
        df = df.dropna(subset=['smiles', 'pchembl_value'])
        df['pchembl_value'] = df['pchembl_value'].astype(float)

        os.makedirs(self.raw_dir, exist_ok=True)
        df.to_csv(self.raw_paths[0], index=False)
        print(f"[ChEMBL] {len(df)} molécules téléchargées → {self.raw_paths[0]}")

    def process(self):
        """
        Convertit les SMILES en graphes PyG et sauvegarde le cache.
        Appelé automatiquement si le fichier traité est absent.
        """
        df = pd.read_csv(self.raw_paths[0])

        # Normalisation des colonnes d'affinité
        if 'pchembl_value' not in df.columns:
            if 'affinite' in df.columns:
                df = df.rename(columns={'affinite': 'pchembl_value'})
            else:
                raise ValueError(
                    "CSV invalide : colonne 'pchembl_value' ou 'affinite' requise."
                )

        data_list = []
        n_invalides = 0

        for _, ligne in df.iterrows():
            smiles = str(ligne['smiles']).strip()
            affinity = float(ligne['pchembl_value'])

            graphe = smiles_vers_graphe(smiles)
            if graphe is None:
                n_invalides += 1
                continue

            # Label d'affinité (scalaire)
            graphe.y = torch.tensor([affinity], dtype=torch.float)
            graphe.smiles = smiles

            if self.pre_transform is not None:
                graphe = self.pre_transform(graphe)

            data_list.append(graphe)

        print(
            f"[Dataset] {len(data_list)} graphes valides | "
            f"{n_invalides} SMILES invalides ignorés."
        )

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


# ---------------------------------------------------------------------------
# Utilitaires de split et DataLoaders
# ---------------------------------------------------------------------------

def creer_dataloaders(
    dataset,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    batch_size: int = 32,
    seed: int = 42,
) -> tuple:
    """
    Divise le dataset et crée les DataLoaders train/val/test.

    Args:
        dataset    : ChEMBLDataset ou liste de Data PyG.
        train_ratio: Fraction allouée à l'entraînement.
        val_ratio  : Fraction allouée à la validation.
        batch_size : Taille des mini-lots.
        seed       : Graine aléatoire pour la reproductibilité.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    n = len(dataset)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val

    generateur = torch.Generator().manual_seed(seed)
    train_set, val_set, test_set = random_split(
        dataset, [n_train, n_val, n_test], generator=generateur
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def creer_dataset_demo(n_molecules: int = 300) -> list:
    """
    Crée un petit dataset synthétique pour tester le pipeline
    sans avoir besoin de données ChEMBL.

    Utilise des médicaments courants et génère des affinités aléatoires
    reproducibles (pKi entre 4 et 10).

    Args:
        n_molecules : Nombre de molécules à générer (par répétition).

    Returns:
        Liste de Data PyG avec attribut y (affinité simulée).
    """
    import random
    random.seed(42)
    torch.manual_seed(42)

    # Médicaments courants avec SMILES valides
    MOLECULES_REFERENCE = [
        'CC(=O)OC1=CC=CC=C1C(=O)O',                          # Aspirine
        'CC(=O)Nc1ccc(O)cc1',                                 # Paracétamol
        'CC(C)Cc1ccc(cc1)C(C)C(=O)O',                        # Ibuprofène
        'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',                     # Caféine
        'OC(=O)c1ccc(N)cc1',                                  # Acide amino-benzoïque
        'c1ccc2ccccc2c1',                                     # Naphtalène
        'CC(C(=O)O)c1ccc2cc(OC)ccc2c1',                     # Naproxène
        'Nc1ccc(cc1)S(N)(=O)=O',                             # Sulfanilamide
        'OC1CCCCC1',                                          # Cyclohexanol
        'CC1=CC=C(C=C1)C(C)(C)C',                            # 4-tert-butyltoluène
        'c1ccc(cc1)O',                                        # Phénol
        'CC(N)Cc1ccc(O)cc1',                                  # Tyrosine méthyl
        'O=C(O)c1cccnc1',                                     # Acide nicotinique
        'Cc1ccc(cc1)S(N)(=O)=O',                             # Toluènesulfonamide
        'OC(=O)CCCC(=O)O',                                    # Acide glutarique
    ]

    data_list = []
    for i in range(n_molecules):
        smiles = MOLECULES_REFERENCE[i % len(MOLECULES_REFERENCE)]
        graphe = smiles_vers_graphe(smiles)
        if graphe is not None:
            # Affinité simulée : pKi entre 4.0 et 10.0
            affinity = 4.0 + 6.0 * random.random()
            graphe.y = torch.tensor([affinity], dtype=torch.float)
            graphe.smiles = smiles
            data_list.append(graphe)

    print(f"[Demo] Dataset synthétique : {len(data_list)} molécules créées.")
    return data_list
