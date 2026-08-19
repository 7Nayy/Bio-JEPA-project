"""
pretrain.py

Phase 1 du protocole de transfert learning Bio-JEPA :
Pré-entraînement NON SUPERVISÉ sur ZINC250k.

──────────────────────────────────────────────────────────────────────────
Protocole complet (Hu et al. 2020 / MolBERT) :

  Phase 1 — Ce script
    → Bio-JEPA apprend des représentations moléculaires génériques
      sur 250 000 molécules drug-like SANS aucun label d'activité.
    → Checkpoint : checkpoints/zinc_pretrained.pt

  Phase 2 — few_shot_eval.py
    → La sonde MLP est entraînée sur N exemples CHEMBL251 labelisés.
    → Le Target Encoder ZINC reste figé.
    → Comparaison avec GNN supervisé from scratch sur les mêmes N exemples.

    python few_shot_eval.py \\
        --checkpoint checkpoints/zinc_pretrained.pt \\
        --target CHEMBL251 \\
        --save-json results/few_shot_transfer.json
──────────────────────────────────────────────────────────────────────────

Différences avec train.py :
  - Dataset : ZINC250k (structurelle, aucun label)
  - Pas d'évaluation par sonde MLP (aucun label disponible)
  - Pas de monitoring Pearson r (labels = dummy 0.0)
  - Affichage : train/val JEPA loss uniquement
  - batch_size = 256 (optimisé pour 250k molécules)

Usage :
    # Pré-entraînement standard (recommandé pour Colab GPU T4)
    python pretrain.py

    # Avec config personnalisée
    python pretrain.py --config configs/pretrain_zinc.yaml --epochs 50

    # Sur Colab
    !python pretrain.py --epochs 100
"""

import argparse

import yaml
import torch
from torch_geometric.loader import DataLoader

from data.zinc_dataset import ZINCDataset
from models.bio_jepa import BioJEPA
from training.trainer import BioJEPATrainer


# ---------------------------------------------------------------------------
# Utilitaires
# ---------------------------------------------------------------------------

def charger_config(chemin: str) -> dict:
    with open(chemin) as f:
        return yaml.safe_load(f)


def afficher_resume_modele(model: BioJEPA):
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


def creer_split_zinc(dataset, train_ratio: float, val_ratio: float,
                     batch_size: int, seed: int):
    """Divise ZINC250k en train/val et crée les DataLoaders."""
    from torch.utils.data import random_split

    n = len(dataset)
    n_train = int(n * train_ratio)
    n_val = n - n_train  # tout le reste → val (pas de test split pour ZINC)

    gen = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=gen)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False,
                              num_workers=2, pin_memory=True)

    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Fonction principale
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Pré-entraînement Bio-JEPA sur ZINC250k (Phase 1 transfert)',
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        '--config', default='configs/pretrain_zinc.yaml',
        help='Config YAML (défaut: configs/pretrain_zinc.yaml)',
    )
    parser.add_argument(
        '--epochs', type=int, default=None,
        help="Nombre d'époques (surcharge la config)",
    )
    args = parser.parse_args()

    config = charger_config(args.config)
    if args.epochs:
        config['entrainement']['num_epochs'] = args.epochs

    cfg_m = config['modele']
    cfg_t = config['entrainement']
    cfg_d = config['donnees']

    print("=" * 70)
    print("  Bio-JEPA — Pré-entraînement sur ZINC250k")
    print("  Protocole de transfert learning (Phase 1 / 2)")
    print("=" * 70)
    print(
        "\n  Après ce script, lancez :\n"
        "    python few_shot_eval.py \\\n"
        "        --checkpoint checkpoints/zinc_pretrained.pt \\\n"
        "        --target CHEMBL251 \\\n"
        "        --save-json results/few_shot_transfer.json\n"
    )

    # --- Chargement de ZINC250k ---
    print("[Données] Chargement de ZINC250k...")
    print("  (Téléchargement automatique au premier appel, ~4 Mo)")

    dataset = ZINCDataset(root=cfg_d['root'])

    train_loader, val_loader = creer_split_zinc(
        dataset,
        train_ratio=cfg_d['train_ratio'],
        val_ratio=cfg_d['val_ratio'],
        batch_size=cfg_t['batch_size'],
        seed=cfg_d['seed'],
    )

    print(
        f"\n  ZINC total    : {len(dataset):,} molécules"
        f"\n  Train         : {len(train_loader.dataset):,} mol."
        f"  │  Val : {len(val_loader.dataset):,} mol."
    )

    # --- Création du modèle ---
    print("\n[Modèle] Architecture Bio-JEPA :")
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

    # --- Pré-entraînement ---
    print(f"\n[Pré-entraînement] Démarrage sur ZINC250k...")
    print(
        f"  Monitoring Pearson r : DÉSACTIVÉ (ZINC = pas de labels)\n"
        f"  Metric de suivi     : JEPA loss (train + val)"
    )

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
        pearson_monitoring=False,   # Pas de labels → pas de Pearson r
    )

    trainer.entrainer(verbose=True)

    # --- Renommage du checkpoint pour la Phase 2 ---
    import os, shutil
    src = os.path.join(config['checkpoints']['dir'], 'best_model.pt')
    dst = os.path.join(config['checkpoints']['dir'], 'zinc_pretrained.pt')
    if os.path.exists(src):
        shutil.copy2(src, dst)
        print(f"\n  Checkpoint Phase 1 sauvegardé → {dst}")

    print("\n" + "=" * 70)
    print("  Pré-entraînement ZINC250k terminé.")
    print("  Lancez maintenant la Phase 2 :")
    print(
        "    python few_shot_eval.py \\\n"
        "        --checkpoint checkpoints/zinc_pretrained.pt \\\n"
        "        --target CHEMBL251 \\\n"
        "        --save-json results/few_shot_transfer.json"
    )
    print("=" * 70)


if __name__ == '__main__':
    main()
