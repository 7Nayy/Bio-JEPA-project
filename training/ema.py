"""
ema.py

Mise à jour par Exponential Moving Average (EMA) du Target Encoder.

Formule :
    θ_target ← τ · θ_target + (1 − τ) · θ_context

où τ (momentum) augmente progressivement de ema_start vers ema_end
selon un cosinus, comme dans BYOL et I-JEPA. Un τ proche de 1
rend le target encoder très stable (évolue lentement).

Les buffers du BatchNorm (running_mean, running_var) sont copiés
directement depuis le context encoder (pas d'EMA sur les statistiques).
"""

import math
import torch
import torch.nn as nn


class EMAUpdater:
    """
    Gestionnaire de la mise à jour EMA du Target Encoder.

    Args:
        ema_start   : Valeur initiale de τ (début d'entraînement).
        ema_end     : Valeur finale de τ (fin d'entraînement).
        total_steps : Nombre total de steps pour le schedule cosinus.
    """

    def __init__(
        self,
        ema_start: float = 0.996,
        ema_end: float = 1.0,
        total_steps: int = 10_000,
    ):
        self.ema_start = ema_start
        self.ema_end = ema_end
        self.total_steps = total_steps
        self.step_actuel = 0

    def get_momentum(self) -> float:
        """
        Calcule le momentum τ à l'étape courante via un schedule cosinus.

        Le momentum augmente de ema_start à ema_end, ce qui ralentit
        progressivement la mise à jour du target encoder au fil de
        l'entraînement (convergence plus stable).

        Returns:
            Valeur de τ ∈ [ema_start, ema_end].
        """
        t = self.step_actuel / max(self.total_steps, 1)
        # Schedule cosinus : lent au début et à la fin
        tau = self.ema_end - (self.ema_end - self.ema_start) * (
            math.cos(math.pi * t) + 1
        ) / 2
        return float(tau)

    @torch.no_grad()
    def update(
        self,
        context_encoder: nn.Module,
        target_encoder: nn.Module,
    ) -> float:
        """
        Met à jour les paramètres du Target Encoder par EMA.

        Appelé après chaque step d'optimisation du Context Encoder.

        Args:
            context_encoder : Encodeur source (entraîné par backprop).
            target_encoder  : Encodeur cible (mis à jour par EMA).

        Returns:
            Valeur de τ utilisée pour cette mise à jour.
        """
        tau = self.get_momentum()

        # --- Mise à jour des paramètres (poids et biais) ---
        for param_c, param_t in zip(
            context_encoder.parameters(),
            target_encoder.parameters(),
        ):
            # EMA : θ_t ← τ·θ_t + (1−τ)·θ_c
            param_t.data.mul_(tau).add_(param_c.data, alpha=1.0 - tau)

        # --- Copie directe des buffers (stats BatchNorm) ---
        # Les running_mean / running_var ne sont pas des paramètres
        # → on les copie directement pour rester cohérent avec le context encoder.
        for buf_c, buf_t in zip(
            context_encoder.buffers(),
            target_encoder.buffers(),
        ):
            buf_t.data.copy_(buf_c.data)

        self.step_actuel += 1
        return tau

    def reset(self):
        """Réinitialise le compteur de steps (ex: pour une nouvelle run)."""
        self.step_actuel = 0
