# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Bio-JEPA adapts the I-JEPA (Joint-Embedding Predictive Architecture) framework for molecular drug discovery — specifically predicting protein-ligand binding affinity (pChEMBL values) from molecular graphs without supervised signal during pre-training.

## Environment Setup

```bash
# Activate the local venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# On macOS Apple Silicon (MPS)
pip install torch torchvision
pip install torch-geometric
pip install rdkit pandas scipy pyyaml tqdm chembl-webresource-client
```

## Common Commands

```bash
# Quick smoke test (synthetic data, no downloads)
python train.py --demo

# Pre-train on ChEMBL251 (downloads automatically, 300 epochs)
python train.py --target CHEMBL251

# Pre-train on ChEMBL251 with custom epochs, skip MLP probe eval
python train.py --target CHEMBL251 --epochs 50 --no-eval

# Pre-train from a local CSV (columns: 'smiles', 'pchembl_value')
python train.py --csv path/to/molecules.csv

# Transfer learning: pre-train on ZINC250k (Phase 1)
python pretrain.py  # uses configs/pretrain_zinc.yaml

# Supervised GNN baseline (same architecture, same splits)
python baseline.py --target CHEMBL251

# Few-shot evaluation: Bio-JEPA frozen encoder vs GNN from scratch
python few_shot_eval.py \
  --checkpoint checkpoints/best_model.pt \
  --target CHEMBL251 \
  --save-json results/few_shot.json

# Transfer few-shot (after pretrain.py on ZINC)
python few_shot_eval.py \
  --checkpoint checkpoints/zinc_pretrained.pt \
  --target CHEMBL251 \
  --save-json results/few_shot_transfer.json
```

## Architecture

### JEPA Training Loop (train.py / pretrain.py → training/trainer.py)
1. **Curriculum masking**: `mask_ratio` increases 10%→30% via cosine schedule over training
2. `context_encoder(masked_graph)` → `z_context` [with gradient]
3. `target_encoder(full_graph)` → `z_target` [EMA updated, no gradient]
4. `predictor(z_context)` → `z_pred`
5. Loss = MSE(`z_pred`, `z_target.detach()`) in L2-normalized latent space

### Key Model Components
- **`models/gnn_encoder.py`**: `GNNEncoder` — 4-layer GINEConv with residual connections, `global_mean_pool + global_max_pool` concatenated → 2× `embedding_dim`
- **`models/predictor.py`**: `Predictor` — MLP with GELU, BatchNorm, Dropout mapping `z_context → z̃_target`
- **`models/bio_jepa.py`**: `BioJEPA` — assembles context encoder (trained), target encoder (EMA copy, frozen), and predictor
- **`training/ema.py`**: `EMAUpdater` — cosine EMA momentum schedule (0.996→1.0), copies BatchNorm buffers
- **`training/trainer.py`**: `BioJEPATrainer` — AdamW + CosineAnnealingLR, Pearson r monitoring via numpy lstsq

### Data Pipeline
- **`data/mol_graph.py`**: SMILES → PyG `Data` object; `ATOM_FEATURE_DIM=29`, `BOND_FEATURE_DIM=7`
- **`data/chembl_dataset.py`**: `ChEMBLDataset` (PyG `InMemoryDataset`) + `creer_dataset_demo()` for synthetic testing
- **`data/zinc_dataset.py`**: `ZINCDataset` for unsupervised pre-training on ZINC250k

### Evaluation
- **`evaluation/metrics.py`**: `SondeMLP` (2-layer MLP probe on frozen embeddings) + `evaluer_sonde_mlp()` + `calculer_metriques()` (Pearson r, Spearman ρ, RMSE, MAE)
- For inference, use `model.encode(batch, encoder='target')` — returns L2-normalized embeddings

### Configs
- `configs/default.yaml`: ChEMBL training (batch=32, 300 epochs)
- `configs/pretrain_zinc.yaml`: ZINC250k pre-training (batch=256, 100 epochs, 95/5 split, no sonde eval)

## Key Dimensions
- `ATOM_FEATURE_DIM = 29` (11 atom type + 6 formal charge + 6 hybridization + 1 aromatic + 5 Hs)
- `BOND_FEATURE_DIM = 7` (5 bond type + 1 conjugated + 1 ring)
- `embedding_dim = 256`, `hidden_dim = 128`, `num_gnn_layers = 4`

## Two Training Protocols

**Protocol 1 — Single-target (default)**:
`train.py` → pre-trains Bio-JEPA on ChEMBL target → MLP probe evaluation

**Protocol 2 — Transfer learning**:
`pretrain.py` on ZINC250k → `few_shot_eval.py` with `--checkpoint zinc_pretrained.pt`

## Device Support
Auto-detected: CUDA → MPS (Apple Silicon) → CPU. No code changes needed.
