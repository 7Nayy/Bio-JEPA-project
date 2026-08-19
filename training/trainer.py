"""
trainer.py

Boucle d'entraînement principale de Bio-JEPA.

Gestion de :
  - L'optimisation du Context Encoder et du Predictor (AdamW).
  - Le schedule cosinus du learning rate.
  - La mise à jour EMA du Target Encoder.
  - Le scheduler de masquage : mask_ratio augmente de mask_ratio_start
    à mask_ratio_end selon un cosinus (curriculum de difficulté).
  - Le Pearson r par époque : calculé rapidement via régression numpy
    (lstsq en O(N·D²), pas de boucle d'optimisation).
  - Le gradient clipping.
  - La sauvegarde des checkpoints.
  - Un affichage ASCII sparkline de la courbe Pearson r en fin d'entraînement.
"""

import math
import os
import time

import numpy as np
import torch
from scipy import stats
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from models.bio_jepa import BioJEPA
from training.ema import EMAUpdater


class BioJEPATrainer:
    """
    Gestionnaire de l'entraînement Bio-JEPA.

    Args:
        model            : Modèle BioJEPA à entraîner.
        train_loader     : DataLoader PyG pour l'entraînement.
        val_loader       : DataLoader PyG pour la validation (optionnel).
        lr               : Learning rate initial d'AdamW.
        weight_decay     : Pénalité L2 sur les paramètres.
        num_epochs       : Nombre d'époques d'entraînement.
        ema_start        : Momentum EMA initial.
        ema_end          : Momentum EMA final.
        gradient_clip    : Norme max du gradient (clipping).
        mask_ratio_start : Fraction d'atomes masqués au début (défaut : 0.10).
        mask_ratio_end   : Fraction d'atomes masqués à la fin  (défaut : 0.30).
        checkpoint_dir   : Dossier de sauvegarde des checkpoints.
        device           : 'cuda', 'mps' ou 'cpu'. Auto-détecté si None.
    """

    def __init__(
        self,
        model: BioJEPA,
        train_loader,
        val_loader=None,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        num_epochs: int = 300,
        ema_start: float = 0.996,
        ema_end: float = 1.0,
        gradient_clip: float = 1.0,
        mask_ratio_start: float = 0.10,
        mask_ratio_end: float = 0.30,
        checkpoint_dir: str = 'checkpoints',
        device: str = None,
        pearson_monitoring: bool = True,
    ):
        # --- Détection automatique du dispositif ---
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            elif torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'

        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_epochs = num_epochs
        self.gradient_clip = gradient_clip
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Paramètres du scheduler de masquage
        self.mask_ratio_start = mask_ratio_start
        self.mask_ratio_end = mask_ratio_end

        # Désactiver pour le pré-entraînement sans labels (ex : ZINC250k)
        self.pearson_monitoring = pearson_monitoring

        # --- Optimiseur (uniquement Context Encoder + Predictor) ---
        params_a_optimiser = (
            list(model.context_encoder.parameters())
            + list(model.predictor.parameters())
        )
        self.optimizer = AdamW(
            params_a_optimiser, lr=lr, weight_decay=weight_decay
        )

        # --- Scheduler cosinus du LR ---
        total_steps = num_epochs * len(train_loader)
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
            eta_min=lr * 0.01,
        )

        # --- Updater EMA ---
        self.ema_updater = EMAUpdater(
            ema_start=ema_start,
            ema_end=ema_end,
            total_steps=total_steps,
        )

        # --- Historique des métriques (une valeur par époque) ---
        self.historique = {
            'train_loss': [],
            'val_loss': [],
            'ema_momentum': [],
            'lr': [],
            'mask_ratio': [],   # fraction de masquage courante
            'pearson_r': [],    # Pearson r sur le val set (sonde linéaire rapide)
        }

    # ------------------------------------------------------------------
    # Scheduler de masquage (curriculum de difficulté)
    # ------------------------------------------------------------------

    def _mise_a_jour_mask_ratio(self, epoch: int) -> float:
        """
        Met à jour model.mask_ratio selon un schedule cosinus croissant.

        Le masquage commence faible (10%) pour que le modèle apprenne facilement,
        puis augmente progressivement jusqu'à 30%, forçant une représentation
        plus robuste et compressive.

        t = 0 (epoch 1)      → mask_ratio_start
        t = 1 (epoch N)      → mask_ratio_end
        Schedule cosinus : montée douce, évite les sauts brusques.

        Args:
            epoch : Numéro de l'époque courante (1-indexé).

        Returns:
            Valeur courante de mask_ratio.
        """
        t = (epoch - 1) / max(self.num_epochs - 1, 1)  # 0.0 → 1.0
        # Cosinus croissant : lent au début et à la fin, plus rapide au milieu
        ratio = self.mask_ratio_start + (
            self.mask_ratio_end - self.mask_ratio_start
        ) * (1 - math.cos(math.pi * t)) / 2

        self.model.mask_ratio = ratio
        return ratio

    # ------------------------------------------------------------------
    # Pearson r rapide (par époque, sans entraîner de sonde)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _pearson_r_rapide(self) -> float:
        """
        Calcule le Pearson r sur le val set via régression numpy (moindres carrés).

        Méthode : np.linalg.lstsq — résolution exacte en O(N·D²), sans boucle
        d'optimisation. Idéale pour un monitoring léger par époque.

        Pipeline :
          1. Forward du Target Encoder sur tout le val set → Z [N, D].
          2. Ajout d'un biais → Z_b [N, D+1].
          3. Résoudre ŷ = Z_b @ w via lstsq sur les données TRAIN (évite le
             leakage val→train de la régression).
          4. Prédire sur val, calculer Pearson r(y_val, ŷ_val).

        Returns:
            Pearson r ∈ [-1, 1] ou nan si val_loader est absent.
        """
        if self.val_loader is None:
            return float('nan')

        self.model.eval()

        # Extraction des embeddings train (pour fitter la régression)
        z_train_list, y_train_list = [], []
        for batch in self.train_loader:
            batch = batch.to(self.device)
            z = self.model.encode(batch, encoder='target')
            z_train_list.append(z.cpu().numpy())
            y_train_list.append(batch.y.cpu().numpy().flatten())

        # Extraction des embeddings val (pour évaluer)
        z_val_list, y_val_list = [], []
        for batch in self.val_loader:
            batch = batch.to(self.device)
            z = self.model.encode(batch, encoder='target')
            z_val_list.append(z.cpu().numpy())
            y_val_list.append(batch.y.cpu().numpy().flatten())

        self.model.train()

        Z_train = np.vstack(z_train_list)                        # [N_train, D]
        y_train = np.concatenate(y_train_list)                   # [N_train]
        Z_val = np.vstack(z_val_list)                            # [N_val, D]
        y_val = np.concatenate(y_val_list)                       # [N_val]

        # Ajout du biais comme colonne de 1
        Z_train_b = np.hstack([Z_train, np.ones((len(Z_train), 1))])
        Z_val_b = np.hstack([Z_val, np.ones((len(Z_val), 1))])

        # Régression moindres carrés (fit sur train)
        w, _, _, _ = np.linalg.lstsq(Z_train_b, y_train, rcond=None)
        y_pred_val = Z_val_b @ w

        # Pearson r sur val
        r, _ = stats.pearsonr(y_val, y_pred_val)
        return float(r)

    # ------------------------------------------------------------------
    # Étapes internes
    # ------------------------------------------------------------------

    def _etape_train(self, batch) -> tuple:
        """
        Effectue une étape d'entraînement sur un mini-lot.

        Returns:
            (loss_val, tau)
        """
        batch = batch.to(self.device)
        self.optimizer.zero_grad()

        loss, _, _, _ = self.model(batch)
        loss.backward()

        params = (
            list(self.model.context_encoder.parameters())
            + list(self.model.predictor.parameters())
        )
        torch.nn.utils.clip_grad_norm_(params, max_norm=self.gradient_clip)

        self.optimizer.step()
        self.scheduler.step()

        tau = self.ema_updater.update(
            self.model.context_encoder,
            self.model.target_encoder,
        )

        return loss.item(), tau

    @torch.no_grad()
    def _etape_validation(self) -> float:
        """Calcule la loss JEPA moyenne sur le jeu de validation."""
        if self.val_loader is None:
            return float('nan')

        self.model.eval()
        total_loss, n_batches = 0.0, 0

        for batch in self.val_loader:
            batch = batch.to(self.device)
            loss, _, _, _ = self.model(batch)
            total_loss += loss.item()
            n_batches += 1

        self.model.train()
        return total_loss / max(n_batches, 1)

    # ------------------------------------------------------------------
    # Affichage sparkline ASCII
    # ------------------------------------------------------------------

    @staticmethod
    def _sparkline(valeurs: list, largeur: int = 60) -> str:
        """
        Génère une sparkline ASCII à partir d'une liste de valeurs.

        Utilise les caractères Unicode de blocs ▁▂▃▄▅▆▇█.
        Les NaN sont représentés par un espace.

        Args:
            valeurs : Liste de floats (une valeur par époque).
            largeur : Nombre de caractères max dans la sparkline.

        Returns:
            Chaîne de caractères représentant l'évolution.
        """
        BLOCS = ' ▁▂▃▄▅▆▇█'
        vals_valides = [v for v in valeurs if v == v]  # exclure NaN
        if not vals_valides:
            return '(aucune valeur)'

        mini, maxi = min(vals_valides), max(vals_valides)
        plage = maxi - mini

        # Sous-échantillonnage si la liste est trop longue
        if len(valeurs) > largeur:
            pas = len(valeurs) / largeur
            valeurs = [valeurs[int(i * pas)] for i in range(largeur)]

        def _bloc(v):
            if v != v:  # NaN
                return ' '
            if plage < 1e-8:
                return BLOCS[4]
            idx = int((v - mini) / plage * 8)
            return BLOCS[min(idx, 8)]

        return ''.join(_bloc(v) for v in valeurs)

    # ------------------------------------------------------------------
    # Boucle principale
    # ------------------------------------------------------------------

    def entrainer(self, verbose: bool = True) -> dict:
        """
        Lance la boucle d'entraînement complète.

        À chaque époque :
          1. Forward/backward sur le train set.
          2. Mise à jour EMA du Target Encoder.
          3. Mise à jour du mask_ratio (curriculum cosinus 10%→30%).
          4. Calcul du Pearson r sur val (via lstsq, rapide).
          5. Affichage de toutes les métriques.

        En fin d'entraînement :
          - Affichage d'une courbe ASCII du Pearson r par époque.

        Args:
            verbose : Affiche les métriques toutes les 5 époques.

        Returns:
            Dictionnaire de l'historique des métriques.
        """
        print(f"{'─'*70}")
        print(f"  Bio-JEPA — Entraînement sur {self.device}")
        print(f"  Époques      : {self.num_epochs}")
        print(f"  Mask ratio   : {self.mask_ratio_start:.0%} → {self.mask_ratio_end:.0%}  (schedule cosinus)")
        print(f"  Train        : {len(self.train_loader.dataset):,} molécules")
        if self.val_loader:
            print(f"  Val          : {len(self.val_loader.dataset):,} molécules")
        print(f"{'─'*70}")

        meilleure_val_loss = float('inf')

        for epoch in range(1, self.num_epochs + 1):
            self.model.train()
            debut = time.time()

            # --- (a) Mise à jour du mask_ratio AVANT les steps ---
            mask_ratio_courant = self._mise_a_jour_mask_ratio(epoch)

            # --- (b) Boucle d'entraînement ---
            losses_epoch, taus_epoch = [], []
            for batch in self.train_loader:
                loss_val, tau = self._etape_train(batch)
                losses_epoch.append(loss_val)
                taus_epoch.append(tau)

            # --- (c) Métriques de l'époque ---
            train_loss = sum(losses_epoch) / len(losses_epoch)
            tau_moy = sum(taus_epoch) / len(taus_epoch)
            val_loss = self._etape_validation()
            lr_actuel = self.scheduler.get_last_lr()[0]

            # --- (d) Pearson r rapide sur val (désactivé si pas de labels) ---
            pearson_r = (
                self._pearson_r_rapide() if self.pearson_monitoring
                else float('nan')
            )

            # --- (e) Enregistrement de l'historique ---
            self.historique['train_loss'].append(train_loss)
            self.historique['val_loss'].append(val_loss)
            self.historique['ema_momentum'].append(tau_moy)
            self.historique['lr'].append(lr_actuel)
            self.historique['mask_ratio'].append(mask_ratio_courant)
            self.historique['pearson_r'].append(pearson_r)

            # --- (f) Sauvegarde du meilleur modèle ---
            val_crit = val_loss if val_loss == val_loss else train_loss
            if val_crit < meilleure_val_loss:
                meilleure_val_loss = val_crit
                self.sauvegarder_checkpoint('best_model.pt')

            # --- (g) Affichage ---
            if verbose and (epoch % 5 == 0 or epoch == 1 or epoch == self.num_epochs):
                duree = time.time() - debut
                val_str = f"{val_loss:.4f}" if val_loss == val_loss else "  ─  "
                r_str = (
                    f"{pearson_r:+.3f}"
                    if (self.pearson_monitoring and pearson_r == pearson_r)
                    else "  ─  "
                )
                r_field = f"  │ r {r_str}" if self.pearson_monitoring else ""
                print(
                    f"  Ep {epoch:4d}/{self.num_epochs}"
                    f"  │ train {train_loss:.4f}"
                    f"  │ val {val_str}"
                    f"{r_field}"
                    f"  │ mask {mask_ratio_courant:.0%}"
                    f"  │ τ {tau_moy:.4f}"
                    f"  │ lr {lr_actuel:.1e}"
                    f"  │ {duree:.1f}s"
                )

        # ------------------------------------------------------------------
        # Résumé final + courbe ASCII du Pearson r
        # ------------------------------------------------------------------
        print(f"\n{'─'*70}")
        print(f"  Entraînement terminé  │  Meilleure val_loss : {meilleure_val_loss:.4f}")
        self.sauvegarder_checkpoint('final_model.pt')

        # Courbe ASCII : Pearson r si monitoring actif, sinon val_loss
        if self.pearson_monitoring:
            rs = self.historique['pearson_r']
            rs_valides = [r for r in rs if r == r]
            if rs_valides:
                sparkline = self._sparkline(rs)
                print(f"\n  Pearson r par époque (val set, sonde linéaire rapide) :")
                print(f"  Ep 1 {'─'*3} Ep {self.num_epochs}")
                print(f"  {sparkline}")
                print(
                    f"  min {min(rs_valides):+.3f}  │  max {max(rs_valides):+.3f}"
                    f"  │  final {rs_valides[-1]:+.3f}"
                )
        else:
            vl = self.historique['val_loss']
            vl_valides = [v for v in vl if v == v]
            if vl_valides:
                sparkline = self._sparkline(vl)
                print(f"\n  Val loss par époque :")
                print(f"  Ep 1 {'─'*3} Ep {self.num_epochs}")
                print(f"  {sparkline}")
                print(
                    f"  début {vl_valides[0]:.4f}  │  fin {vl_valides[-1]:.4f}"
                    f"  │  min {min(vl_valides):.4f}"
                )

        print(f"{'─'*70}\n")
        return self.historique

    # ------------------------------------------------------------------
    # Checkpoints
    # ------------------------------------------------------------------

    def sauvegarder_checkpoint(self, nom_fichier: str):
        """Sauvegarde l'état complet du modèle et de l'optimiseur."""
        chemin = os.path.join(self.checkpoint_dir, nom_fichier)
        torch.save(
            {
                'context_encoder': self.model.context_encoder.state_dict(),
                'target_encoder': self.model.target_encoder.state_dict(),
                'predictor': self.model.predictor.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
                'historique': self.historique,
                'ema_step': self.ema_updater.step_actuel,
                'mask_ratio': self.model.mask_ratio,
            },
            chemin,
        )

    def charger_checkpoint(self, nom_fichier: str):
        """Charge un checkpoint et restaure l'état complet."""
        chemin = os.path.join(self.checkpoint_dir, nom_fichier)
        ckpt = torch.load(chemin, map_location=self.device, weights_only=False)

        self.model.context_encoder.load_state_dict(ckpt['context_encoder'])
        self.model.target_encoder.load_state_dict(ckpt['target_encoder'])
        self.model.predictor.load_state_dict(ckpt['predictor'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.scheduler.load_state_dict(ckpt['scheduler'])
        self.historique = ckpt.get('historique', self.historique)
        self.ema_updater.step_actuel = ckpt.get('ema_step', 0)
        self.model.mask_ratio = ckpt.get('mask_ratio', self.mask_ratio_end)

        print(f"  Checkpoint chargé : {chemin}")
