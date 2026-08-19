"""
umap_viz.py

Visualisation UMAP des embeddings Bio-JEPA sur ChEMBL251.

Pipeline :
  1. Charge checkpoints/zinc_pretrained.pt
  2. Extrait les embeddings (Target Encoder figé) de tout ChEMBL251
  3. Réduit en 2D avec UMAP (métrique cosine, adapté aux embeddings L2-normalisés)
  4. Génère deux figures dans results/ :
       - umap_affinity.png  : colormap viridis par pChEMBL
       - umap_scaffold.png  : top 10 scaffolds de Murcko + gris pour les autres
  5. Identifie les 5 clusters les plus denses (KMeans) et affiche leurs
     molécules représentatives (plus proche du centroïde)

Dépendances supplémentaires :
    pip install umap-learn matplotlib

Usage :
    python umap_viz.py
    python umap_viz.py --checkpoint checkpoints/best_model.pt
    python umap_viz.py --n-clusters 30 --top-scaffolds 12
"""

import argparse
import os
from collections import Counter

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from torch_geometric.loader import DataLoader

from data.chembl_dataset import ChEMBLDataset
from evaluation.metrics import extraire_embeddings
from models.bio_jepa import BioJEPA


# ── Chargement du modèle ──────────────────────────────────────────────────────

def charger_modele(checkpoint_path: str, device: torch.device) -> BioJEPA:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = BioJEPA()
    if isinstance(ckpt, dict) and 'target_encoder' in ckpt:
        model.context_encoder.load_state_dict(ckpt['context_encoder'])
        model.target_encoder.load_state_dict(ckpt['target_encoder'])
        model.predictor.load_state_dict(ckpt['predictor'])
    else:
        model.load_state_dict(ckpt)
    return model.to(device).eval()


# ── Scaffolds de Murcko ───────────────────────────────────────────────────────

def calculer_scaffolds(smiles_list: list) -> list:
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    out = []
    for smi in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                out.append('INVALID')
            else:
                sc = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
                out.append(sc if sc else 'NO_SCAFFOLD')
        except Exception:
            out.append('INVALID')
    return out


# ── Analyse de clusters ───────────────────────────────────────────────────────

def analyser_clusters(
    coords_2d: np.ndarray,
    embeddings: np.ndarray,
    smiles_list: list,
    labels: np.ndarray,
    n_clusters: int = 30,
    top_k: int = 5,
) -> list:
    """
    KMeans sur les coordonnées 2D UMAP, sélection des top_k clusters par taille.

    Pour chaque cluster retourne :
      - centroid_2d    : coordonnées 2D du centroïde
      - size           : nombre de molécules
      - rep_idx        : indice global de la molécule représentative
      - rep_smiles     : son SMILES
      - rep_pchembl    : son score pChEMBL
      - mean_pchembl   : pChEMBL moyen du cluster
      - std_pchembl    : écart-type pChEMBL du cluster
    """
    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    cluster_ids = km.fit_predict(coords_2d)

    # Sélection des top_k clusters par taille
    compteur = Counter(cluster_ids)
    top_ids = [c for c, _ in compteur.most_common(top_k)]

    infos = []
    for cid in top_ids:
        masque = cluster_ids == cid
        indices = np.where(masque)[0]

        centroid_2d = coords_2d[masque].mean(axis=0)

        # Représentant : plus proche du centroïde dans l'espace 2D
        dists = np.linalg.norm(coords_2d[masque] - centroid_2d, axis=1)
        local_best = np.argmin(dists)
        rep_idx = indices[local_best]

        infos.append({
            'cluster_id'   : cid,
            'centroid_2d'  : centroid_2d,
            'size'         : int(masque.sum()),
            'rep_idx'      : int(rep_idx),
            'rep_smiles'   : smiles_list[rep_idx],
            'rep_pchembl'  : float(labels[rep_idx]),
            'mean_pchembl' : float(labels[masque].mean()),
            'std_pchembl'  : float(labels[masque].std()),
        })

    return infos


def annoter_clusters(ax, clusters: list):
    """Dessine les centroïdes et numéros de clusters sur un axe matplotlib."""
    for rank, cl in enumerate(clusters, start=1):
        cx, cy = cl['centroid_2d']
        ax.scatter(cx, cy, s=180, marker='*', color='white',
                   edgecolors='black', linewidths=0.8, zorder=10)
        ax.text(
            cx, cy + 0.6, str(rank),
            ha='center', va='bottom', fontsize=9, fontweight='bold',
            color='white',
            path_effects=[pe.withStroke(linewidth=2, foreground='black')],
            zorder=11,
        )


# ── Figure A — Affinité pChEMBL ───────────────────────────────────────────────

def figure_affinite(
    coords_2d: np.ndarray,
    labels: np.ndarray,
    clusters: list,
    out_path: str,
    checkpoint_name: str,
):
    fig, ax = plt.subplots(figsize=(10, 8))

    sc = ax.scatter(
        coords_2d[:, 0], coords_2d[:, 1],
        c=labels, cmap='viridis',
        vmin=labels.min(), vmax=labels.max(),
        s=7, alpha=0.75, linewidths=0, rasterized=True,
    )

    cbar = plt.colorbar(sc, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label('pChEMBL (affinité de liaison)', fontsize=11)
    cbar.ax.tick_params(labelsize=9)

    annoter_clusters(ax, clusters)

    ax.set_title(
        f'Bio-JEPA — UMAP des embeddings moléculaires\n'
        f'Coloré par affinité pChEMBL  ·  checkpoint : {checkpoint_name}'
        f'  ·  N = {len(labels):,}',
        fontsize=12,
    )
    ax.set_xlabel('UMAP 1', fontsize=10)
    ax.set_ylabel('UMAP 2', fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [A] {out_path}")


# ── Figure B — Scaffolds de Murcko ────────────────────────────────────────────

def figure_scaffolds(
    coords_2d: np.ndarray,
    scaffolds: list,
    clusters: list,
    out_path: str,
    checkpoint_name: str,
    top_n: int = 10,
):
    INVALIDES = {'INVALID', 'NO_SCAFFOLD', ''}
    compteur = Counter(s for s in scaffolds if s not in INVALIDES)
    top_scaffolds = [s for s, _ in compteur.most_common(top_n)]
    sc2idx = {s: i for i, s in enumerate(top_scaffolds)}

    n_col = max(len(top_scaffolds), 2)
    palette = plt.get_cmap('tab20')(np.linspace(0, 1, n_col))
    gris = (0.80, 0.80, 0.80, 0.35)

    fig, ax = plt.subplots(figsize=(14, 8))

    # Autres scaffolds en arrière-plan
    masque_autre = np.array([s not in sc2idx for s in scaffolds])
    if masque_autre.any():
        ax.scatter(
            coords_2d[masque_autre, 0], coords_2d[masque_autre, 1],
            color=gris, s=5, linewidths=0, rasterized=True, zorder=1,
        )

    # Top N scaffolds
    handles = []
    for i, scaffold in enumerate(top_scaffolds):
        masque = np.array([s == scaffold for s in scaffolds])
        label = scaffold[:36] + '…' if len(scaffold) > 36 else scaffold
        ax.scatter(
            coords_2d[masque, 0], coords_2d[masque, 1],
            color=palette[i], s=16, alpha=0.88, linewidths=0,
            rasterized=True, zorder=2 + i,
        )
        handles.append(
            Line2D([0], [0], marker='o', color='w', markerfacecolor=palette[i],
                   markersize=7, label=f'{label}  (n={masque.sum()})')
        )

    n_autres = masque_autre.sum()
    handles.append(
        Line2D([0], [0], marker='o', color='w', markerfacecolor=gris[:3],
               alpha=0.7, markersize=7, label=f'Autres / acycliques  (n={n_autres})')
    )

    annoter_clusters(ax, clusters)

    ax.legend(
        handles=handles,
        title=f'Top {top_n} scaffolds de Murcko',
        bbox_to_anchor=(1.01, 1), loc='upper left',
        fontsize=7, title_fontsize=9,
        framealpha=0.95, borderpad=0.9,
    )

    ax.set_title(
        f'Bio-JEPA — UMAP des embeddings moléculaires\n'
        f'Coloré par scaffold de Murcko  ·  checkpoint : {checkpoint_name}'
        f'  ·  {len(compteur):,} scaffolds uniques  ·  N = {len(scaffolds):,}',
        fontsize=12,
    )
    ax.set_xlabel('UMAP 1', fontsize=10)
    ax.set_ylabel('UMAP 2', fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [B] {out_path}")


# ── Affichage des clusters ─────────────────────────────────────────────────────

def afficher_clusters(clusters: list):
    largeur = 70
    print("\n" + "═" * largeur)
    print("  5 CLUSTERS LES PLUS DENSES (KMeans sur UMAP 2D)")
    print("═" * largeur)
    for rank, cl in enumerate(clusters, start=1):
        smi = cl['rep_smiles']
        smi_court = smi[:55] + '…' if len(smi) > 55 else smi
        print(
            f"\n  #{rank}  taille={cl['size']:>4}  "
            f"pChEMBL moy={cl['mean_pchembl']:.2f} ± {cl['std_pchembl']:.2f}"
        )
        print(f"       Représentant (pChEMBL={cl['rep_pchembl']:.2f}) :")
        print(f"       SMILES : {smi_court}")
    print("\n" + "═" * largeur)


# ── Point d'entrée ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Visualisation UMAP des embeddings Bio-JEPA',
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        '--checkpoint', default='checkpoints/zinc_pretrained.pt',
        help='Checkpoint Bio-JEPA (défaut: checkpoints/zinc_pretrained.pt)',
    )
    parser.add_argument('--target', default='CHEMBL251')
    parser.add_argument('--csv', default=None)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--top-scaffolds', type=int, default=10)
    parser.add_argument('--umap-neighbors', type=int, default=15)
    parser.add_argument('--umap-min-dist', type=float, default=0.1)
    parser.add_argument('--n-clusters', type=int, default=30,
                        help='Nombre de clusters KMeans pour trouver les zones denses')
    parser.add_argument('--out-dir', default='results')
    args = parser.parse_args()

    # --- Dépendances optionnelles ---
    try:
        import umap as umap_lib
    except ImportError:
        raise ImportError("Installez umap-learn :  pip install umap-learn")

    # --- Dispositif ---
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"[Dispositif] {device}")

    os.makedirs(args.out_dir, exist_ok=True)
    checkpoint_name = os.path.splitext(os.path.basename(args.checkpoint))[0]

    # --- Modèle ---
    print(f"\n[Modèle] {args.checkpoint}")
    if not os.path.exists(args.checkpoint):
        # Fallback sur best_model.pt si zinc_pretrained.pt absent
        fallback = 'checkpoints/best_model.pt'
        if os.path.exists(fallback):
            print(f"  zinc_pretrained.pt absent → fallback sur {fallback}")
            args.checkpoint = fallback
            checkpoint_name = 'best_model'
        else:
            raise FileNotFoundError(
                f"Checkpoint introuvable : {args.checkpoint}\n"
                "Lancez d'abord :  python pretrain.py  ou  python train.py --target CHEMBL251"
            )
    model = charger_modele(args.checkpoint, device)
    print(f"  Target Encoder : {sum(p.numel() for p in model.target_encoder.parameters()):,} paramètres")

    # --- Dataset ---
    print(f"\n[Données] ChEMBL — target={args.target}")
    dataset = ChEMBLDataset(
        root='data/chembl',
        csv_path=args.csv,
        target_chembl_id=args.target,
    )
    print(f"  {len(dataset):,} molécules")

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # --- Embeddings ---
    print("\n[Embeddings] Extraction via Target Encoder...")
    embeddings, labels = extraire_embeddings(model, loader, device, encoder='target')
    print(f"  Shape : {embeddings.shape}")
    print(f"  pChEMBL : min={labels.min():.2f}  max={labels.max():.2f}"
          f"  mean={labels.mean():.2f}  std={labels.std():.2f}")

    # --- Scaffolds ---
    print("\n[Scaffolds] Calcul Murcko...")
    smiles_list = [dataset[i].smiles for i in range(len(dataset))]
    scaffolds = calculer_scaffolds(smiles_list)
    n_uniques = len({s for s in scaffolds if s not in {'INVALID', 'NO_SCAFFOLD', ''}})
    print(f"  {n_uniques:,} scaffolds uniques")

    # --- UMAP ---
    print(f"\n[UMAP] {embeddings.shape[1]}D → 2D"
          f"  (n_neighbors={args.umap_neighbors}  min_dist={args.umap_min_dist})")
    reducer = umap_lib.UMAP(
        n_neighbors=args.umap_neighbors,
        min_dist=args.umap_min_dist,
        n_components=2,
        metric='cosine',
        random_state=42,
        verbose=True,
    )
    coords_2d = reducer.fit_transform(embeddings)
    print(f"  Réduction terminée → {coords_2d.shape}")

    # --- Clusters ---
    print(f"\n[Clusters] KMeans (k={args.n_clusters}), sélection top 5...")
    clusters = analyser_clusters(
        coords_2d, embeddings, smiles_list, labels,
        n_clusters=args.n_clusters, top_k=5,
    )
    afficher_clusters(clusters)

    # --- Figures ---
    print("\n[Figures] Génération...")
    figure_affinite(
        coords_2d, labels, clusters,
        out_path=os.path.join(args.out_dir, 'umap_affinity.png'),
        checkpoint_name=checkpoint_name,
    )
    figure_scaffolds(
        coords_2d, scaffolds, clusters,
        out_path=os.path.join(args.out_dir, 'umap_scaffold.png'),
        checkpoint_name=checkpoint_name,
        top_n=args.top_scaffolds,
    )

    print(f"\nFait. Figures dans {args.out_dir}/")


if __name__ == '__main__':
    main()
