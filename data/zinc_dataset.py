"""
data/zinc_dataset.py

Dataset ZINC250k pour le pré-entraînement non supervisé de Bio-JEPA.

ZINC250k est un sous-ensemble de ~250 000 molécules drug-like de la base ZINC,
couramment utilisé comme corpus de pré-entraînement dans les papiers de
représentation moléculaire (Hu et al. 2020, MolBERT, ChemBERTa).

Protocole de transfert learning :
  Phase 1 — Pré-entraînement non supervisé sur ZINC250k (ce dataset).
             → Aucun label d'activité requis.
             → Bio-JEPA apprend des représentations moléculaires génériques.

  Phase 2 — Fine-tuning few-shot sur CHEMBL251 (via few_shot_eval.py).
             → La sonde MLP est entraînée sur N exemples labelisés.
             → Le Target Encoder reste figé.

Le dataset est téléchargé automatiquement au premier appel (~4 Mo).
Les graphes PyG sont mis en cache dans <root>/processed/data.pt.
"""

import os
import urllib.request

import pandas as pd
import torch
from torch_geometric.data import InMemoryDataset

from data.mol_graph import smiles_vers_graphe


class ZINCDataset(InMemoryDataset):
    """
    Dataset ZINC250k pour le pré-entraînement Bio-JEPA.

    Télécharge automatiquement le CSV ZINC250k (250 000 SMILES)
    et le convertit en graphes moléculaires PyG.

    Attributs PyG créés :
        data.x          : features atomiques [num_atomes, 29]
        data.edge_index : connectivité des liaisons [2, num_liaisons]
        data.edge_attr  : features des liaisons [num_liaisons, 7]
        data.y          : dummy tensor([0.0]) — JEPA est auto-supervisé,
                          aucun label de propriété n'est utilisé durant
                          le pré-entraînement.

    Args:
        root          : Répertoire de cache. Crée <root>/raw/ et <root>/processed/.
        transform     : Transformation PyG appliquée à chaque graphe (optionnel).
        pre_transform : Transformation appliquée avant la mise en cache (optionnel).
    """

    # CSV ZINC250k depuis le dépôt chemical_vae (Gómez-Bombarelli et al. 2018)
    # ~4 Mo, 250 000 SMILES avec logP, QED, SAS
    URL = (
        'https://raw.githubusercontent.com/aspuru-guzik-group/'
        'chemical_vae/master/models/zinc_properties/'
        '250k_rndm_zinc_drugs_clean_3.csv'
    )
    NOM_CSV = '250k_rndm_zinc_drugs_clean_3.csv'

    def __init__(
        self,
        root: str = 'data/zinc',
        transform=None,
        pre_transform=None,
    ):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(
            self.processed_paths[0], weights_only=False
        )

    # ------------------------------------------------------------------
    # Interface InMemoryDataset
    # ------------------------------------------------------------------

    @property
    def raw_file_names(self) -> list:
        return [self.NOM_CSV]

    @property
    def processed_file_names(self) -> list:
        return ['data.pt']

    # ------------------------------------------------------------------
    # Téléchargement
    # ------------------------------------------------------------------

    def download(self):
        """Télécharge le CSV ZINC250k depuis GitHub."""
        dest = os.path.join(self.raw_dir, self.NOM_CSV)
        print(f"[ZINC] Téléchargement de ZINC250k depuis GitHub...")
        print(f"  URL  : {self.URL}")
        print(f"  Dest : {dest}")

        try:
            urllib.request.urlretrieve(self.URL, dest)
            taille_mo = os.path.getsize(dest) / 1e6
            print(f"  ✓ Téléchargement réussi ({taille_mo:.1f} Mo)")
        except Exception as e:
            raise RuntimeError(
                f"Échec du téléchargement ZINC250k : {e}\n"
                "Téléchargez manuellement le CSV depuis :\n"
                f"  {self.URL}\n"
                f"et placez-le dans : {self.raw_dir}"
            ) from e

    # ------------------------------------------------------------------
    # Traitement : SMILES → graphes PyG
    # ------------------------------------------------------------------

    def process(self):
        """Convertit le CSV ZINC250k en graphes PyG et met en cache."""
        csv_path = os.path.join(self.raw_dir, self.NOM_CSV)
        df = pd.read_csv(csv_path)

        # La colonne SMILES peut s'appeler 'smiles' ou 'SMILES'
        if 'smiles' in df.columns:
            col_smiles = 'smiles'
        elif 'SMILES' in df.columns:
            col_smiles = 'SMILES'
        else:
            raise ValueError(
                f"Colonne SMILES introuvable dans {csv_path}. "
                f"Colonnes disponibles : {list(df.columns)}"
            )

        smiles_list = df[col_smiles].dropna().tolist()
        print(f"[ZINC] {len(smiles_list):,} SMILES lus → conversion en graphes...")

        data_list = []
        n_invalides = 0

        for i, smiles in enumerate(smiles_list):
            if i % 50_000 == 0 and i > 0:
                print(f"  {i:,}/{len(smiles_list):,} traités...")

            graphe = smiles_vers_graphe(smiles)
            if graphe is None:
                n_invalides += 1
                continue

            # Label dummy : Bio-JEPA est auto-supervisé, aucune propriété
            # moléculaire n'est utilisée durant le pré-entraînement.
            graphe.y = torch.tensor([0.0], dtype=torch.float)
            data_list.append(graphe)

        print(
            f"[ZINC] Conversion terminée : "
            f"{len(data_list):,} graphes valides │ "
            f"{n_invalides} SMILES invalides ignorés "
            f"({n_invalides / len(smiles_list) * 100:.1f}%)"
        )

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        print(f"[ZINC] Graphes mis en cache → {self.processed_paths[0]}")
