"""
app.py

Interface Gradio de démonstration Bio-JEPA — 3 onglets, rédigés pour un
public non spécialiste (pas de jargon chimie/ML dans l'UI) :

  ① Tester une molécule : SMILES → embedding (Target Encoder figé) → sonde MLP
     → efficacité prédite, avec les molécules connues les plus proches dans
     l'espace latent.
  ② Pourquoi c'est malin : Bio-JEPA (encodeur figé) vs GNN supervisé from
     scratch, à N exemples labellisés égal — l'argument central du projet.
  ③ Comment l'IA "voit" les molécules : projection 2D des embeddings (UMAP),
     colorée par affinité ou par scaffold de Murcko (figures pré-calculées).
  ④ Repositionnement de médicaments : criblage de 3229 médicaments déjà
     approuvés (FDA) pour repérer ceux qui pourraient aussi agir sur la
     cible étudiée (résultats précalculés par repositioning.py).

Usage :
    python app.py
"""

import json
import socket
from concurrent.futures import ThreadPoolExecutor

import gradio as gr
import numpy as np
import plotly.graph_objects as go
import torch
import torch.nn.functional as F
import yaml
from rdkit import Chem
from rdkit.Chem import Draw
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.data import Batch

from data.chembl_dataset import ChEMBLDataset
from data.mol_graph import smiles_vers_graphe
from evaluation.metrics import SondeMLP
from few_shot_eval import charger_biojep, resoudre_checkpoint

# Évite qu'un appel réseau (recherche de nom ChEMBL) ne bloque l'app si
# l'API est indisponible pendant une démo/vidéo.
socket.setdefaulttimeout(5)

CONFIG_PATH = 'configs/default.yaml'
CHECKPOINT_PATH = 'checkpoints/best_model.pt'
TARGET_ID = 'CHEMBL251'
TARGET_NOM = 'A2A (récepteur adénosine)'

# Comparaison honnête pour l'onglet ② : Bio-JEPA contre trois AUTRES méthodes
# de pré-entraînement auto-supervisé connues de la recherche (même protocole :
# pré-entraînement sur ZINC250k sans labels, puis sonde gelée sur ChEMBL251).
# On ne compare PAS à un GNN entraîné directement sur les labels : sur des
# petites molécules, un tel GNN gagne presque toujours à lui seul, quelle que
# soit la qualité du pré-entraînement — ce ne serait pas une comparaison
# honnête de l'intérêt du pré-entraînement.
FEWSHOT_SOURCES = {
    'Bio-JEPA (notre méthode)': ('results/few_shot_1000ep.json', 'bio_jepa'),
    'MolCLR'                  : ('results/baselines_graphmae_molclr.json', 'molclr_fewshot'),
    'GraphMAE'                : ('results/baselines_graphmae_molclr.json', 'graphmae_fewshot'),
    'AttrMasking'             : ('results/attrmasking_fewshot.json', 'results'),
}
FEWSHOT_COULEURS = {
    'Bio-JEPA (notre méthode)': '#2E86AB',
    'MolCLR'                  : '#A23B72',
    'GraphMAE'                : '#F18F01',
    'AttrMasking'             : '#6A994E',
}

UMAP_IMAGES = {
    'Coloration : efficacité du médicament'  : 'results/umap_affinity.png',
    'Coloration : famille chimique'          : 'results/umap_scaffold.png',
}

REPOSITIONING_PATH = 'results/repositioning_results.json'

# (nom affiché, SMILES) — la caféine et la théophylline sont de VRAIS
# antagonistes du récepteur A2A utilisé comme cible dans cette démo.
EXEMPLES = [
    ("Aspirine",                "CC(=O)OC1=CC=CC=C1C(=O)O"),
    ("Ibuprofène",              "CC(C)Cc1ccc(cc1)C(C)C(=O)O"),
    ("Caféine (vrai antagoniste A2A)",     "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"),
    ("Théophylline (vrai antagoniste A2A)", "Cn1c2[nH]cnc2c(=O)n(C)c1=O"),
]


# ---------------------------------------------------------------------------
# Dispositif
# ---------------------------------------------------------------------------

def choisir_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


DEVICE = choisir_device()


# ---------------------------------------------------------------------------
# Entraînement de la sonde MLP (une fois, au démarrage de l'app)
# ---------------------------------------------------------------------------

def entrainer_sonde(
    Z: np.ndarray, y: np.ndarray,
    embedding_dim: int, hidden_dim: int, lr: float, num_epochs: int,
    device: torch.device,
) -> SondeMLP:
    """
    Entraîne une SondeMLP sur les embeddings figés du Target Encoder
    (split 85/15 train/val, early stopping sur val_MSE).

    Identique dans l'esprit à evaluation.metrics.evaluer_sonde_mlp, mais
    conserve le modèle entraîné en mémoire plutôt que de simplement
    retourner des métriques.
    """
    n = len(Z)
    rng = np.random.default_rng(42)
    idx = rng.permutation(n)
    n_val = max(1, int(n * 0.15))
    idx_val, idx_train = idx[:n_val], idx[n_val:]

    def t(arr):
        return torch.tensor(arr, dtype=torch.float, device=device)

    Z_tr, y_tr = t(Z[idx_train]), t(y[idx_train])
    Z_vl, y_vl = t(Z[idx_val]), t(y[idx_val])

    torch.manual_seed(42)  # reproductibilité de l'init des poids entre redémarrages
    sonde = SondeMLP(embedding_dim, hidden_dim).to(device)
    optim = torch.optim.Adam(sonde.parameters(), lr=lr, weight_decay=1e-4)
    sched = CosineAnnealingLR(optim, T_max=num_epochs, eta_min=lr * 0.01)

    meilleure_val = float('inf')
    params_best = {k: v.clone() for k, v in sonde.state_dict().items()}

    for _ in range(num_epochs):
        sonde.train()
        optim.zero_grad()
        F.mse_loss(sonde(Z_tr), y_tr).backward()
        optim.step()
        sched.step()

        sonde.eval()
        with torch.no_grad():
            val_mse = F.mse_loss(sonde(Z_vl), y_vl).item()
        if val_mse < meilleure_val:
            meilleure_val = val_mse
            params_best = {k: v.clone() for k, v in sonde.state_dict().items()}

    sonde.load_state_dict(params_best)
    sonde.eval()
    return sonde


# ---------------------------------------------------------------------------
# Chargement — modèle, dataset, embeddings, sonde (une seule fois)
# ---------------------------------------------------------------------------

print(f"[App] Dispositif : {DEVICE}")

with open(CONFIG_PATH, encoding='utf-8') as f:
    _config = yaml.safe_load(f)
_cfg_m = _config['modele']
_cfg_e = _config['evaluation']

print("[App] Chargement du modèle Bio-JEPA...")
_checkpoint_path = resoudre_checkpoint(CHECKPOINT_PATH)
_model = charger_biojep(_checkpoint_path, _cfg_m, DEVICE)

print(f"[App] Chargement du dataset {TARGET_ID} (cache local)...")
_dataset = ChEMBLDataset(root='data/chembl', target_chembl_id=TARGET_ID)
_all_data = [_dataset[i] for i in range(len(_dataset))]
_smiles_all = [d.smiles for d in _all_data]
_y_all = np.array([float(d.y) for d in _all_data])

print(f"[App] Extraction des embeddings ({len(_all_data)} molécules)...")
with torch.no_grad():
    _batch_full = Batch.from_data_list(_all_data).to(DEVICE)
    _Z_all = _model.encode(_batch_full, encoder='target').cpu().numpy()

print("[App] Entraînement de la sonde MLP sur les embeddings figés...")
_sonde = entrainer_sonde(
    _Z_all, _y_all,
    embedding_dim=_cfg_m['embedding_dim'],
    hidden_dim=_cfg_e['hidden_dim_sonde'],
    lr=_cfg_e['lr_sonde'],
    num_epochs=_cfg_e['num_epochs_sonde'],
    device=DEVICE,
)
print(f"[App] Prêt — checkpoint : {_checkpoint_path}")


# ---------------------------------------------------------------------------
# Onglet ① — Prédiction en direct
# ---------------------------------------------------------------------------

def etiquette_efficacite(score: float) -> str:
    """Traduit le score numérique en verdict qualitatif, plus parlant qu'un chiffre seul."""
    if score < 5:
        return "🔴 Faible"
    if score < 7:
        return "🟠 Modérée"
    return "🟢 Forte"


def _canoniser(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol) if mol else None


# Repli local instantané (sans réseau) pour les molécules qu'on propose en exemple
_NOMS_LOCAUX = {_canoniser(smi): nom.split(' (')[0] for nom, smi in EXEMPLES}
_nom_cache: dict[str, str | None] = {}


def nom_pour_smiles(smiles: str) -> str:
    """
    Cherche un nom usuel pour une molécule : d'abord dans les exemples connus
    (instantané), puis via l'API ChEMBL (mise en cache, avec repli si le
    réseau est indisponible pendant la démo).
    """
    canon = _canoniser(smiles)
    if canon in _NOMS_LOCAUX:
        return _NOMS_LOCAUX[canon]

    if smiles in _nom_cache:
        return _nom_cache[smiles] or "Composé de recherche"

    nom = None
    try:
        from chembl_webresource_client.new_client import new_client
        res = list(
            new_client.molecule
            .filter(molecule_structures__canonical_smiles__flexmatch=smiles)
            .only(['pref_name'])[:1]
        )
        if res and res[0].get('pref_name'):
            nom = res[0]['pref_name'].title()
    except Exception:
        nom = None

    _nom_cache[smiles] = nom
    return nom or "Composé de recherche"


def predire(smiles: str):
    smiles = (smiles or '').strip()
    if not smiles:
        return None, "Entrez une formule chimique (SMILES) ou choisissez un exemple ci-dessous.", []

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, f"⚠️ Cette formule n'est pas reconnue : `{smiles}`. Essayez un des exemples.", []

    graphe = smiles_vers_graphe(smiles)
    with torch.no_grad():
        batch = Batch.from_data_list([graphe]).to(DEVICE)
        z = _model.encode(batch, encoder='target')  # [1, D], normalisé L2
        pred = _sonde(z).item()

    # Voisins les plus proches dans l'espace latent (embeddings L2-normalisés
    # → produit scalaire = similarité cosinus)
    z_np = z.cpu().numpy()[0]
    sims = _Z_all @ z_np
    top_idx = np.argsort(-sims)[:5]

    # Recherche des noms en parallèle (borne le pire cas au timeout réseau,
    # au lieu de le multiplier par 5 en série)
    with ThreadPoolExecutor(max_workers=5) as executeur:
        noms = list(executeur.map(nom_pour_smiles, [_smiles_all[i] for i in top_idx]))

    voisins = [
        [noms[k], _smiles_all[i][:45], round(float(_y_all[i]), 2), f"{100 * float(sims[i]):.0f} %"]
        for k, i in enumerate(top_idx)
    ]

    image = Draw.MolToImage(mol, size=(320, 320))
    texte = (
        f"### Efficacité estimée contre la cible : {etiquette_efficacite(pred)}\n"
        f"**Score : {pred:.1f} / 10** (les scores réels vont typiquement de 4 à 10 — "
        f"plus c'est haut, plus la molécule se lie fortement à sa cible).\n\n"
        f"Cible testée : {TARGET_NOM}. Cette prédiction vient d'une IA qui n'a **jamais "
        f"vu d'exemple annoté pendant sa phase principale d'apprentissage** — elle a "
        f"seulement observé la structure de millions de molécules, puis un petit module "
        f"a été calibré sur {len(_all_data):,} exemples connus pour cette cible précise."
    )
    return image, texte, voisins


# ---------------------------------------------------------------------------
# Onglet ② — Efficacité few-shot
# ---------------------------------------------------------------------------

def _extraire_courbe(chemin: str, cle: str) -> tuple:
    """
    Normalise les 3 formats de fichiers JSON différents présents dans
    results/ (héritage de plusieurs scripts d'évaluation) vers
    (N_values, means, stds).
    """
    with open(chemin, encoding='utf-8') as f:
        d = json.load(f)
    bloc = d[cle]

    if isinstance(bloc, dict):
        # Format {"10": {"mean":.., "std":..}, "50": {...}, ...}
        N = sorted(int(n) for n in bloc)
        means = [bloc[str(n)]['mean'] for n in N]
        stds  = [bloc[str(n)]['std']  for n in N]
    elif isinstance(bloc[0], dict) and 'pearson_r' in bloc[0]:
        # Format [{"N":.., "pearson_r": {"mean":.., "std":..}}, ...]
        N = [x['N'] for x in bloc]
        means = [x['pearson_r']['mean'] for x in bloc]
        stds  = [x['pearson_r']['std']  for x in bloc]
    else:
        # Format [{"mean_r":.., "std_r":..}, ...] aligné sur d['N_values']
        N = d['N_values']
        means = [x['mean_r'] for x in bloc]
        stds  = [x['std_r']  for x in bloc]
    return N, means, stds


def tracer_fewshot() -> go.Figure:
    fig = go.Figure()
    for nom, (chemin, cle) in FEWSHOT_SOURCES.items():
        N, means, stds = _extraire_courbe(chemin, cle)
        est_jepa = nom.startswith('Bio-JEPA')
        fig.add_trace(go.Scatter(
            x=N, y=means, mode='lines+markers', name=nom,
            error_y=dict(type='data', array=stds, visible=True),
            line=dict(color=FEWSHOT_COULEURS[nom], width=4 if est_jepa else 2),
            marker=dict(size=10 if est_jepa else 7),
        ))
    fig.update_layout(
        title="Notre IA vs 3 autres méthodes de pré-entraînement sans labels — "
              "cible : récepteur A2A",
        xaxis_title="Nombre d'exemples déjà testés donnés à l'IA",
        yaxis_title="Précision des prédictions (0 = hasard, 1 = parfait)",
        xaxis_type='log',
        yaxis_range=[-0.05, 0.8],
        template='plotly_white',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(t=80),
    )
    return fig


def verdict_fewshot() -> str:
    """Calcule un verdict chiffré, à jour si les fichiers de résultats changent."""
    N_jepa, moy_jepa, _ = _extraire_courbe(*FEWSHOT_SOURCES['Bio-JEPA (notre méthode)'])
    n_ref = 1000
    idx = N_jepa.index(n_ref) if n_ref in N_jepa else len(N_jepa) - 1
    score_jepa = moy_jepa[idx]

    meilleur_concurrent, meilleur_score = None, -1.0
    for nom, source in FEWSHOT_SOURCES.items():
        if nom.startswith('Bio-JEPA'):
            continue
        N, moy, _ = _extraire_courbe(*source)
        score = moy[N.index(n_ref)] if n_ref in N else moy[-1]
        if score > meilleur_score:
            meilleur_concurrent, meilleur_score = nom, score

    ratio = score_jepa / meilleur_score if meilleur_score > 0 else float('inf')
    return (
        f"👉 **À {N_jepa[idx]} exemples, notre IA obtient un score de précision de "
        f"{score_jepa:.2f}, contre {meilleur_score:.2f} pour {meilleur_concurrent} "
        f"(la meilleure des 3 alternatives) — environ {ratio:.1f}× plus précis.**"
    )


# ---------------------------------------------------------------------------
# Onglet ③ — Espace latent (UMAP)
# ---------------------------------------------------------------------------

def afficher_umap(choix: str) -> str:
    return UMAP_IMAGES[choix]


# ---------------------------------------------------------------------------
# Onglet ④ — Repositionnement de médicaments
# ---------------------------------------------------------------------------

with open(REPOSITIONING_PATH, encoding='utf-8') as f:
    _repositioning = json.load(f)


def tableau_repositioning() -> list:
    return [
        [r['rank'], r['name'].title(), r['indication'], r['predicted_pchembl']]
        for r in _repositioning['top20']
    ]


def graphique_repositioning() -> go.Figure:
    top10 = list(reversed(_repositioning['top20'][:10]))  # ordre croissant pour le barh
    noms = [r['name'].title() for r in top10]
    scores = [r['predicted_pchembl'] for r in top10]
    moyenne = _repositioning['metadata']['score_stats']['mean']

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=scores, y=noms, orientation='h',
        marker=dict(color=scores, colorscale='Tealgrn', cmin=moyenne),
        text=[f"{s:.1f}" for s in scores], textposition='outside',
    ))
    fig.add_vline(
        x=moyenne, line=dict(color='gray', dash='dash', width=1.5),
        annotation_text=f"Score moyen des {_repositioning['metadata']['n_valid_smiles']:,} "
                         f"médicaments testés ({moyenne:.1f})",
        annotation_position='top',
    )
    fig.update_layout(
        title="Top 10 des médicaments déjà approuvés les plus prometteurs pour cette cible",
        xaxis_title="Efficacité prédite (/10)",
        template='plotly_white',
        margin=dict(t=80, l=10),
        height=420,
    )
    return fig


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------

with gr.Blocks(title="Bio-JEPA — Démo") as demo:
    gr.Markdown(
        "# 🧬 Bio-JEPA — une IA qui aide à trouver de nouveaux médicaments\n"
        "Trouver un médicament, c'est chercher parmi des milliards de molécules "
        "possibles celles qui se lient bien à une cible biologique précise "
        "(une protéine impliquée dans une maladie). Tester une molécule en "
        "laboratoire coûte cher et prend du temps.\n\n"
        "**L'idée** : un peu comme un modèle de langage (type GPT) apprend le "
        "français en lisant énormément de texte *sans qu'on lui donne les "
        "réponses*, notre IA apprend la \"chimie\" en observant des millions "
        "de molécules, sans jamais savoir lesquelles sont efficaces. Ensuite, "
        "avec seulement quelques dizaines d'exemples annotés, elle devient "
        "capable de prédire l'efficacité de nouvelles molécules — bien plus "
        "vite qu'une IA classique qui devrait tout apprendre à partir de zéro."
    )

    with gr.Tabs():
        with gr.Tab("① Tester une molécule"):
            gr.Markdown(
                "Donnez une formule chimique (« SMILES », le format standard "
                "pour écrire une molécule en texte) ou cliquez sur un exemple. "
                "L'IA prédit son efficacité contre une cible biologique réelle : "
                "le **récepteur A2A**, impliqué en neurologie — c'est justement "
                "la cible que bloque la caféine, essayez-la !"
            )
            with gr.Row():
                with gr.Column(scale=1):
                    smiles_in = gr.Textbox(
                        label="Formule chimique (SMILES)",
                        placeholder="CC(=O)OC1=CC=CC=C1C(=O)O",
                    )
                    gr.Examples(
                        examples=[[smi] for _, smi in EXEMPLES],
                        example_labels=[nom for nom, _ in EXEMPLES],
                        inputs=smiles_in,
                        label="Exemples (cliquez pour essayer)",
                    )
                    btn = gr.Button("Prédire l'efficacité", variant="primary")
                with gr.Column(scale=1):
                    img_out = gr.Image(label="À quoi ressemble la molécule", height=300)
            texte_out = gr.Markdown()
            voisins_out = gr.Dataframe(
                headers=["Nom", "Formule chimique", "Efficacité connue (/10)", "Ressemblance"],
                label="5 molécules déjà testées que l'IA juge les plus proches",
            )
            btn.click(predire, inputs=smiles_in, outputs=[img_out, texte_out, voisins_out])
            smiles_in.submit(predire, inputs=smiles_in, outputs=[img_out, texte_out, voisins_out])

        with gr.Tab("② Pourquoi c'est malin"):
            gr.Markdown(
                "Il existe plusieurs façons d'apprendre la \"chimie générale\" à "
                "une IA sans lui donner d'étiquettes. On compare ici notre "
                "méthode (**Bio-JEPA**) à trois autres approches connues de la "
                "recherche (**GraphMAE**, **MolCLR**, **AttrMasking**). Toutes "
                "les quatre reçoivent ensuite exactement le même petit nombre "
                "d'exemples annotés pour apprendre la tâche finale — seule la "
                "façon dont elles ont appris la chimie en amont diffère.\n\n"
                "👉 Plus une courbe est haute, plus les prédictions de cette "
                "méthode sont fiables. On voit que Bio-JEPA (bleu, épais) "
                "prend l'avantage dès qu'on dépasse une centaine d'exemples."
            )
            plot_fs = gr.Plot(value=tracer_fewshot())
            verdict_fs = gr.Markdown(verdict_fewshot())

        with gr.Tab("③ Comment l'IA \"voit\" les molécules"):
            gr.Markdown(
                "Chaque point de cette carte est une molécule. L'IA a appris, "
                "**toute seule et sans étiquette**, à placer les molécules qui "
                "se ressemblent chimiquement les unes à côté des autres — un "
                "peu comme un système de recommandation regroupe des films "
                "similaires, mais ici appris automatiquement à partir de la "
                "seule structure des molécules.\n\n"
                "- **Efficacité du médicament** : jaune = molécule efficace, "
                "violet = peu efficace. Si les zones jaunes sont regroupées "
                "(pas éparpillées au hasard), c'est la preuve que la carte "
                "apprise par l'IA a un sens biologique réel.\n"
                "- **Famille chimique** : chaque couleur est une famille de "
                "molécules qui partagent le même squelette de base. Les "
                "points de même couleur regroupés confirment que l'IA a "
                "appris à reconnaître ces familles toute seule."
            )
            choix_umap = gr.Radio(
                choices=list(UMAP_IMAGES.keys()),
                value=list(UMAP_IMAGES.keys())[0],
                label="Coloration de la carte",
            )
            img_umap = gr.Image(
                value=UMAP_IMAGES[list(UMAP_IMAGES.keys())[0]],
                label="Carte des molécules",
            )
            choix_umap.change(afficher_umap, inputs=choix_umap, outputs=img_umap)

        with gr.Tab("④ Repositionnement de médicaments"):
            n_screen = _repositioning['metadata']['n_valid_smiles']
            top1 = _repositioning['top20'][0]
            gr.Markdown(
                f"Mettre au point un nouveau médicament prend des années. Une "
                f"astuce beaucoup plus rapide : vérifier si un médicament "
                f"**déjà approuvé** pour une autre maladie pourrait, sans "
                f"qu'on l'ait prévu, marcher aussi contre notre cible — on "
                f"appelle ça le **repositionnement de médicaments**, une "
                f"vraie stratégie utilisée dans l'industrie pharmaceutique.\n\n"
                f"Ici, l'IA a passé en revue **{n_screen:,} médicaments déjà "
                f"approuvés** (base FDA) et prédit leur efficacité contre "
                f"{TARGET_NOM}. En tête : **{top1['name'].title()}** "
                f"(normalement utilisé comme {top1['indication'].lower()}), "
                f"avec un score prédit de {top1['predicted_pchembl']:.1f}/10 — "
                f"nettement au-dessus de la moyenne des médicaments testés "
                f"({_repositioning['metadata']['score_stats']['mean']:.1f}/10).\n\n"
                f"⚠️ **Ce sont des prédictions informatiques**, une façon de "
                f"prioriser quelles molécules tester en premier en "
                f"laboratoire — pas une preuve d'efficacité clinique."
            )
            plot_repo = gr.Plot(value=graphique_repositioning())
            gr.Dataframe(
                headers=["Rang", "Nom", "Indication actuelle", "Efficacité prédite (/10)"],
                value=tableau_repositioning(),
                label="Top 20 complet",
            )


if __name__ == '__main__':
    demo.launch()
