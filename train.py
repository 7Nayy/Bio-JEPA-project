"""
train.py

Point d'entrée principal pour l'entraînement de Bio-JEPA.

Usage :
    # Mode démonstration (dataset synthétique, pas de données requises)
    python train.py --demo

    # Avec un CSV local (colonnes : 'smiles', 'pchembl_value')
    python train.py --csv path/to/molecules.csv

    # Téléchargement automatique depuis ChEMBL (nécessite chembl-webresource-client)
    python train.py --target CHEMBL251

    # Avec une config personnalisée
    python train.py --demo --config configs/default.yaml --epochs 50

Sur Google Colab :
    !python train.py --demo --epochs 30
"""

import argparse

import yaml
import torch
from torch_geometric.loader import DataLoader

from data.chembl_dataset import (
    ChEMBLDataset,
    creer_dataloaders,
    creer_dataset_demo,
)
from models.bio_jepa import BioJEPA
from training.trainer import BioJEPATrainer
from evaluation.metrics import evaluer_sonde_mlp


# ---------------------------------------------------------------------------
# Utilitaires
# ---------------------------------------------------------------------------

def charger_config(chemin: str) -> dict:
    """Charge la configuration depuis un fichier YAML."""
    with open(chemin) as f:
        return yaml.safe_load(f)


def afficher_resume_modele(model: BioJEPA):
    """Affiche le nombre de paramètres entraînables."""
    n_context = sum(
        p.numel() for p in model.context_encoder.parameters() if p.requires_grad
    )
    n_predictor = sum(
        p.numel() for p in model.predictor.parameters() if p.requires_grad
    )
    n_target = sum(p.numel() for p in model.target_encoder.parameters())
    print(f"  Context Encoder : {n_context:>10,} paramètres (entraînables)")
    print(f"  Predictor       : {n_predictor:>10,} paramètres (entraînables)")
    print(f"  Target Encoder  : {n_target:>10,} paramètres (EMA, figés)")
    print(f"  Total entraînable : {n_context + n_predictor:,}")


# ---------------------------------------------------------------------------
# Fonction principale
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Entraînement Bio-JEPA — Drug Discovery avec JEPA',
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        '--config', default='configs/default.yaml',
        help='Chemin vers le fichier de configuration YAML (défaut: configs/default.yaml)'
    )
    parser.add_argument(
        '--demo', action='store_true',
        help='Utilise un dataset synthétique (aucune donnée requise)'
    )
    parser.add_argument(
        '--csv', type=str, default=None,
        help="Chemin vers un CSV avec colonnes 'smiles' et 'pchembl_value'"
    )
    parser.add_argument(
        '--target', type=str, default=None,
        help="ID de cible ChEMBL pour le téléchargement automatique (ex: CHEMBL251)"
    )
    parser.add_argument(
        '--epochs', type=int, default=None,
        help='Nombre d\'époques (surcharge la config)'
    )
    parser.add_argument(
        '--no-eval', action='store_true',
        help='Saute l\'évaluation par sonde linéaire'
    )
    args = parser.parse_args()

    # --- Chargement de la configuration ---
    config = charger_config(args.config)
    if args.epochs:
        config['entrainement']['num_epochs'] = args.epochs

    cfg_m = config['modele']
    cfg_t = config['entrainement']
    cfg_d = config['donnees']
    cfg_e = config['evaluation']

    print("=" * 60)
    print("  Bio-JEPA — Joint-Embedding Predictive Architecture")
    print("  pour la Drug Discovery (affinité protéine-ligand)")
    print("=" * 60)

    # --- Chargement des données ---
    if args.demo:
        print("\n[Données] Mode démonstration (dataset synthétique)")
        data_list = creer_dataset_demo(n_molecules=500)

        # Split manuel reproductible
        n = len(data_list)
        n_train = int(n * cfg_d['train_ratio'])
        n_val = int(n * cfg_d['val_ratio'])

        torch.manual_seed(cfg_d['seed'])
        indices = torch.randperm(n).tolist()

        train_data = [data_list[i] for i in indices[:n_train]]
        val_data = [data_list[i] for i in indices[n_train:n_train + n_val]]
        test_data = [data_list[i] for i in indices[n_train + n_val:]]

        bs = cfg_t['batch_size']
        train_loader = DataLoader(train_data, batch_size=bs, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=bs, shuffle=False)
        test_loader = DataLoader(test_data, batch_size=bs, shuffle=False)

    else:
        print("\n[Données] Chargement du dataset ChEMBL...")
        dataset = ChEMBLDataset(
            root=cfg_d['root'],
            csv_path=args.csv,
            target_chembl_id=args.target or 'CHEMBL251',
        )
        train_loader, val_loader, test_loader = creer_dataloaders(
            dataset,
            train_ratio=cfg_d['train_ratio'],
            val_ratio=cfg_d['val_ratio'],
            batch_size=cfg_t['batch_size'],
            seed=cfg_d['seed'],
        )

    print(
        f"\n  Train : {len(train_loader.dataset):,} mol."
        f"  │  Val : {len(val_loader.dataset):,} mol."
        f"  │  Test : {len(test_loader.dataset):,} mol."
    )

    # --- Création du modèle ---
    print("\n[Modèle] Architecture Bio-JEPA :")
    # Le modèle démarre avec mask_ratio_start ; le trainer le fait évoluer
    model = BioJEPA(
        hidden_dim=cfg_m['hidden_dim'],
        embedding_dim=cfg_m['embedding_dim'],
        num_gnn_layers=cfg_m['num_gnn_layers'],
        predictor_hidden_dim=cfg_m['predictor_hidden_dim'],
        predictor_num_layers=cfg_m['predictor_num_layers'],
        mask_ratio=cfg_m['mask_ratio_start'],
        dropout=cfg_m['dropout'],
    )
    afficher_resume_modele(model)

    # --- Entraînement ---
    print(f"\n[Entraînement] Démarrage...")
    trainer = BioJEPATrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=cfg_t['lr'],
        weight_decay=cfg_t['weight_decay'],
        num_epochs=cfg_t['num_epochs'],
        ema_start=cfg_t['ema_start'],
        ema_end=cfg_t['ema_end'],
        gradient_clip=cfg_t['gradient_clip'],
        mask_ratio_start=cfg_m['mask_ratio_start'],
        mask_ratio_end=cfg_m['mask_ratio_end'],
        checkpoint_dir=config['checkpoints']['dir'],
    )

    historique = trainer.entrainer(verbose=True)

    # --- Évaluation par sonde MLP ---
    if not args.no_eval:
        print("\n[Évaluation] Sonde MLP 2 couches sur le Target Encoder figé...")
        metriques = evaluer_sonde_mlp(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=trainer.device,
            embedding_dim=cfg_m['embedding_dim'],
            hidden_dim=cfg_e['hidden_dim_sonde'],
            lr=cfg_e['lr_sonde'],
            num_epochs=cfg_e['num_epochs_sonde'],
            verbose=True,
        )

        print("\n" + "=" * 60)
        print("  Résultats finaux (test set)")
        print("=" * 60)
        for nom, val in metriques.items():
            print(f"  {nom:<16} : {val:.4f}")
    else:
        print("\n[Évaluation] Sautée (--no-eval).")

    print("\n  Checkpoints sauvegardés dans :", config['checkpoints']['dir'])
    print("  Entraînement terminé.")


if __name__ == '__main__':
    main()
