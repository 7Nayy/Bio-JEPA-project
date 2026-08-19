# Bio-JEPA

Learning semantic molecular representations for drug repositioning via protein-ligand affinity prediction, with quantum fingerprint validation.

Bio-JEPA adapts the Joint-Embedding Predictive Architecture (JEPA) to molecular graphs: instead of reconstructing masked atoms/bonds, a Context Encoder predicts the latent representation produced by an EMA-updated Target Encoder for the full molecule, under a stop-gradient. Pre-trained without labels on 249,455 ZINC250k molecules, the frozen encoder is evaluated with a linear/MLP probe on protein-ligand affinity (ChEMBL251/203/325).

Full write-up: [`JEPA_x_BioMedical.pdf`](JEPA_x_BioMedical.pdf) (final results). Initial thesis proposal: [`TD3.pdf`](TD3.pdf).

## Key results

- r = 0.787 (p < 0.001) on ChEMBL251 (A2A receptor) with no affinity labels seen during pre-training.
- Ablation: removing the stop-gradient drops r by 0.222 — the single most critical component.
- The only SSL baseline (vs. GraphMAE, MolCLR, AttrMasking) that scales continuously with labeled data (r = 0.036 → 0.542 for N = 10 → 1000 under a fair, matched-budget protocol).
- ×153,735 speedup over AutoDock Vina on real experimental measurements (50 molecules, receptor 3EML).
- Screened 3,229 FDA-approved drugs for A2A repositioning in under 30 minutes; top candidates cross-checked against a hybrid AI-quantum (Pulser/QuTiP neutral-atom) fingerprint.

See `results/` for the raw metrics (JSON) and figures behind these numbers.

## Repository layout

```
models/          GNNEncoder (GINEConv), Predictor, BioJEPA (assembles context/target/predictor)
training/        EMAUpdater (cosine momentum 0.996→1.0), BioJEPATrainer
evaluation/      MLP probe + metrics (Pearson r, Spearman ρ, RMSE, MAE)
data/            SMILES → graph conversion, ChEMBL/ZINC dataset loaders, raw ChEMBL CSV
configs/         Training configs (default.yaml, pretrain_zinc.yaml)

train.py                 Single-target training (Protocol 1)
pretrain.py               Unsupervised pre-training on ZINC250k (Protocol 2)
few_shot_eval.py          Few-shot linear/MLP probe evaluation
baseline.py                Supervised GNN baseline (same architecture/splits)
ablation_study.py          Stop-gradient / EMA / masking ablations
autodock_benchmark.py      Bio-JEPA vs. AutoDock Vina timing/accuracy comparison
repositioning.py           FDA drug repositioning screening
umap_viz.py                Latent space UMAP + scaffold clustering

notebooks/       Colab orchestration notebooks (pre-training, baselines, few-shot
                 eval, AutoDock comparison, UMAP/repositioning, ablations)
checkpoints/     Pre-trained weights and cached molecular graphs
results/         Evaluation outputs (JSON metrics, UMAP figures)

JEPA_x_BioMedical.pdf   Full paper (NeurIPS format)
TD3.pdf                 Initial thesis proposal
CLAUDE.md               Architecture/commands reference
```

The notebooks in `notebooks/` were run on Google Colab (A100 GPU): each installs
dependencies, pulls the relevant checkpoint(s) from Google Drive, runs one phase
of the pipeline against the scripts above, and saves results back to Drive.

## Checkpoints

| File | Description | Size |
|---|---|---|
| `zinc_pretrained.pt` | Bio-JEPA encoder, 100 epochs on ZINC250k | 13 MB |
| `zinc_pretrained_1000ep.pt` | Same, 1000 epochs (near-identical results, see ablation) | 13 MB |
| `best_model.pt` | Best checkpoint from Phase 1/2 pre-training (used throughout the paper's experiments) | 13 MB |
| `attrmasking_pretrained.pt` / `graphmae_pretrained.pt` / `molclr_pretrained.pt` | SSL baselines, same backbone/budget | ~6 MB each |
| `chembl251_graphs.pt` / `chembl325_graphs.pt` | Cached molecular graphs for evaluation | ~50–60 MB |
| `zinc_graphs.pt.gz` | Cached ZINC250k graphs (gzip'd; ~88 MB vs. 1.4 GB raw) | 88 MB |

All `.pt` / `.pt.gz` files are tracked with **Git LFS**. `zinc_graphs.pt.gz` must be decompressed before use:

```bash
gunzip -k checkpoints/zinc_graphs.pt.gz
```

`data/chembl/processed/data.pt` (PyG cache) is not versioned — it's regenerated
automatically from `data/raw/chembl_data.csv` the first time `data/chembl_dataset.py`
runs.

## Reproducing

```bash
pip install -r requirements.txt

# Phase 1 — unsupervised pre-training on ZINC250k
python pretrain.py

# Phase 2 — few-shot evaluation of the frozen encoder
python few_shot_eval.py --checkpoint checkpoints/zinc_pretrained.pt --target CHEMBL251

# Supervised GNN baseline for comparison
python baseline.py --target CHEMBL251

# FDA drug repositioning screening
python repositioning.py --checkpoint checkpoints/zinc_pretrained.pt --target CHEMBL251 --top-k 20 --out-dir results/
```

See [`CLAUDE.md`](CLAUDE.md) for the full command reference and architecture notes.
