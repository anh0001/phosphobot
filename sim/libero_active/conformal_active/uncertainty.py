"""Conformal action uncertainty for VLA rollouts.

The pipeline computes a per-timestep scalar uncertainty from a VLA policy and
calibrates it so the *query rate* matches a target level. Two layers:

1. Raw dispersion — sample K action chunks from the (stochastic / flow-matching)
   policy and measure self-disagreement. This is the nonconformity signal.
2. Conformal calibration — map raw scores to a calibrated decision so that, under
   exchangeability, the long-run query rate ~= target_alpha. Optionally adapt the
   threshold online with Intermittent Quantile Tracking (IQT) to stay valid as the
   policy is fine-tuned between rounds (distribution drift).

References
- Vovk, Gammerman, Shafer (2005), Algorithmic Learning in a Random World — split conformal.
- Gibbs & Candes (2021), Adaptive Conformal Inference Under Distribution Shift — the
  alpha_{t+1} = alpha_t + gamma * (alpha - err_t) online update.
- Chua et al. (2025), Conformalized Interactive Imitation Learning (ConformalDAgger,
  arXiv:2410.08852) — intermittent-label quantile tracking.

All functions are numpy-based and have no LeRobot/LIBERO dependency, so they are
unit-testable in isolation (see tests at the bottom of run_sweep.py / a dedicated test).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def raw_dispersion(action_samples: np.ndarray) -> float:
    """Self-disagreement of a policy across K sampled action chunks.

    Args:
        action_samples: array (K, T, A) — K sampled chunks, horizon T, action dim A.
            For a deterministic policy pass K identical samples (dispersion == 0).

    Returns:
        Mean per-(timestep, dim) standard deviation across the K samples. Higher ==
        the policy disagrees with itself == more uncertain.
    """
    arr = np.asarray(action_samples, dtype=np.float64)
    if arr.ndim != 3:
        raise ValueError(f"expected (K, T, A), got shape {arr.shape}")
    if arr.shape[0] < 2:
        return 0.0
    return float(arr.std(axis=0).mean())


def _conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample-corrected (1 - alpha) conformal quantile of calibration scores.

    Uses the standard ceil((n+1)(1-alpha))/n rank so that, for exchangeable data, a
    fresh score exceeds the quantile with probability <= alpha.
    """
    n = len(scores)
    if n == 0:
        return float("inf")
    rank = int(np.ceil((n + 1) * (1.0 - alpha)))
    rank = min(max(rank, 1), n)  # clamp into [1, n]
    return float(np.sort(scores)[rank - 1])


@dataclass
class IQTTracker:
    """Adaptive Conformal Inference with intermittent labels.

    Maintains an effective miscoverage level `alpha_t`. After each timestep where a
    ground-truth label is available (an expert correction was or was not needed), call
    `update(was_uncertain_correct)`. Steps with no label leave `alpha_t` unchanged —
    this is the "intermittent" part from ConformalDAgger.

    The query threshold is then the (1 - alpha_t) conformal quantile of recent scores.
    """

    target_alpha: float
    gamma: float = 0.05
    alpha_t: float = field(init=False)

    def __post_init__(self) -> None:
        self.alpha_t = self.target_alpha

    def update(self, miscovered: bool) -> None:
        """One ACI step. `miscovered` == the label fell outside the prediction set."""
        err = 1.0 if miscovered else 0.0
        self.alpha_t += self.gamma * (self.target_alpha - err)
        # Keep alpha_t a usable probability; clamp away from the degenerate ends.
        self.alpha_t = float(np.clip(self.alpha_t, 1e-3, 0.999))

    @property
    def effective_alpha(self) -> float:
        return self.alpha_t


@dataclass
class ConformalActionUncertainty:
    """Calibrated action-uncertainty signal for the conformal query method.

    Workflow per active-learning round:
      1. `reset_calibration()` and feed held-out calibration scores via `add_calibration`.
      2. During rollout, call `score(action_samples)` -> raw float, and
         `is_uncertain(raw)` -> bool query decision.
      3. When an episode resolves (success/failure known), call `observe_label` so IQT
         can adapt `alpha_t`.

    The calibration set must be recollected every round because LoRA fine-tuning shifts
    the policy's score distribution (exchangeability is only round-local).
    """

    target_alpha: float = 0.1
    gamma: float = 0.05
    calib_min: int = 20
    use_iqt: bool = True

    _calib: list[float] = field(default_factory=list, init=False)
    _iqt: IQTTracker = field(init=False)

    def __post_init__(self) -> None:
        self._iqt = IQTTracker(target_alpha=self.target_alpha, gamma=self.gamma)

    # -- calibration -------------------------------------------------------
    def reset_calibration(self) -> None:
        self._calib.clear()
        self._iqt = IQTTracker(target_alpha=self.target_alpha, gamma=self.gamma)

    def add_calibration(self, raw_score: float) -> None:
        self._calib.append(float(raw_score))

    @property
    def is_calibrated(self) -> bool:
        return len(self._calib) >= self.calib_min

    @property
    def threshold(self) -> float:
        """Current query threshold on the raw score scale."""
        alpha = self._iqt.effective_alpha if self.use_iqt else self.target_alpha
        return _conformal_quantile(np.asarray(self._calib), alpha)

    # -- scoring -----------------------------------------------------------
    def score(self, action_samples: np.ndarray) -> float:
        """Raw nonconformity score for the current step."""
        return raw_dispersion(action_samples)

    def is_uncertain(self, raw_score: float) -> bool:
        """Query decision. Before calibration, never query (avoid spurious triggers)."""
        if not self.is_calibrated:
            return False
        return raw_score > self.threshold

    def normalized(self, raw_score: float) -> float:
        """Raw score expressed as a multiple of the threshold (>1 == query)."""
        thr = self.threshold
        if not np.isfinite(thr) or thr <= 0:
            return 0.0
        return raw_score / thr

    # -- online adaptation -------------------------------------------------
    def observe_label(self, miscovered: bool) -> None:
        """Feed an intermittent label to IQT (no-op when use_iqt is False)."""
        if self.use_iqt:
            self._iqt.update(miscovered)

    @property
    def effective_alpha(self) -> float:
        return self._iqt.effective_alpha if self.use_iqt else self.target_alpha
