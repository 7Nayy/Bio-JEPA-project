# Bio-JEPA

Learning semantic molecular representations for drug repositioning via protein-ligand affinity prediction, with quantum fingerprint validation.

Bio-JEPA adapts the Joint-Embedding Predictive Architecture (JEPA) to molecular graphs: instead of reconstructing masked atoms/bonds, a Context Encoder predicts the latent representation produced by an EMA-updated Target Encoder for the full molecule, under a stop-gradient. Pre-trained without labels on 249,455 ZINC250k molecules, the frozen encoder is evaluated with a linear/MLP probe on protein-ligand affinity (ChEMBL251/203/325).

Full write-up: [`JEPA_x_BioMedical.pdf`](JEPA_x_BioMedical.pdf).

## Key results

- r = 0.787 (p < 0.001) on ChEMBL251 (A2A receptor) with no affinity labels seen during pre-training.
- Ablation: removing the stop-gradient drops r by 0.222 — the single most critical component.
- The only SSL baseline (vs. GraphMAE, MolCLR, AttrMasking) that scales continuously with labeled data (r = 0.036 → 0.542 for N = 10 → 1000 under a fair, matched-budget protocol).
- ×153,735 speedup over AutoDock Vina on real experimental measurements (50 molecules, receptor 3EML).
- Screened 3,229 FDA-approved drugs for A2A repositioning in under 30 minutes; top candidates cross-checked against a hybrid AI-quantum (Pulser/QuTiP neutral-atom) fingerprint.

See `Bio-JEPA-results/` for the raw metrics (JSON) and figures behind these numbers.

## Repository layout

```
Bio-Jepa/                  Colab notebooks (pre-training, baselines, few-shot eval,
                            AutoDock comparison, UMAP/repositioning, ablations)
Bio-JEPA-checkpoints/      Pre-trained weights and cached molecular graphs
Bio-JEPA-results/          Evaluation outputs (JSON metrics, UMAP figures)
JEPA_x_BioMedical.pdf      Full paper (NeurIPS format)
```

The training/evaluation source code that the notebooks pull in at runtime
(`train.py`, `pretrain.py`, `few_shot_eval.py`, `repositioning.py`, `umap_viz.py`,
`models/`) is versioned separately; each notebook clones it at the start of its
first cell. This repository holds the experiment orchestration, trained
artifacts, and results.

## Checkpoints

| File | Description | Size |
|---|---|---|
| `zinc_pretrained.pt` | Bio-JEPA encoder, 100 epochs on ZINC250k | 13 MB |
| `zinc_pretrained_1000ep.pt` | Same, 1000 epochs (near-identical results, see ablation) | 13 MB |
| `best_model.pt` | Best checkpoint from Phase 1/2 pre-training | 13 MB |
| `attrmasking_pretrained.pt` / `graphmae_pretrained.pt` / `molclr_pretrained.pt` | SSL baselines, same backbone/budget | ~6 MB each |
| `chembl251_graphs.pt` / `chembl325_graphs.pt` | Cached molecular graphs for evaluation | ~50–60 MB |
| `zinc_graphs.pt.gz` | Cached ZINC250k graphs (gzip'd; ~88 MB vs. 1.4 GB raw) | 88 MB |

All `.pt` / `.pt.gz` files are tracked with **Git LFS**. `zinc_graphs.pt.gz` must be decompressed before use:

```bash
gunzip -k Bio-JEPA-checkpoints/zinc_graphs.pt.gz
```

## Reproducing

Each notebook in `Bio-Jepa/` is self-contained for Google Colab (A100 GPU recommended): it installs dependencies, clones the source code repo, pulls the relevant checkpoint(s) from Google Drive, runs the corresponding phase, and saves results back to Drive.
