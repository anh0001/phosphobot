"""Demo-query strategies for the active-learning sweep (PS.3 / PS.6).

Five methods share one interface so the main-results sweep can compare them fairly:

    random        — uniform random scoring (the default everyone uses today)
    entropy       — action-sample dispersion, thresholded by a running quantile
                    (uncalibrated — the cheap uncertainty baseline)
    knn           — kNN distance to already-collected demo states in feature space
                    (CRSAIL-style coverage / novelty)
    human_gated   — sim proxy for "a human would intervene here": uses privileged
                    oracle-vs-policy disagreement (the honest real-world competitor)
    conformal     — our method: conformally-calibrated action uncertainty + IQT

Fair-comparison protocol: every method assigns a scalar `score` to each rollout step
(higher == more want a corrective demo here). The active loop ranks candidate steps and
spends a shared per-round demo budget on the top-scored ones. `random` just emits random
scores. This keeps the demo budget identical across methods — the only thing that varies
is *which* states get demonstrated.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import ConformalConfig, QueryMethodName
from .uncertainty import ConformalActionUncertainty, raw_dispersion


@dataclass
class StepContext:
    """Per-timestep information available to a query method during a rollout."""

    step: int
    action_samples: np.ndarray            # (K, T, A) sampled action chunks
    feature: np.ndarray | None = None     # (D,) visual/state embedding for kNN
    oracle_disagreement: float | None = None  # privileged: ||oracle_a - policy_a||


class QueryMethod:
    """Base class. Subclasses implement `score`; the loop ranks scores under a budget."""

    name: QueryMethodName

    def reset_round(self) -> None:
        """Called at the start of each active-learning round."""

    def score(self, ctx: StepContext) -> float:
        raise NotImplementedError

    def register_demo_state(self, feature: np.ndarray) -> None:
        """Called when a demo is collected at a state (kNN uses this; others ignore)."""

    def observe_episode(self, *, miscovered: bool) -> None:
        """Intermittent label after an episode resolves (conformal/IQT uses this)."""


class RandomQuery(QueryMethod):
    name = "random"

    def __init__(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)

    def score(self, ctx: StepContext) -> float:
        return float(self._rng.random())


class EntropyQuery(QueryMethod):
    """Uncalibrated dispersion. Same raw signal as conformal but with NO conformal
    calibration and NO IQT — isolates the value of calibration in ablation."""

    name = "entropy"

    def score(self, ctx: StepContext) -> float:
        return raw_dispersion(ctx.action_samples)


class KNNCoverageQuery(QueryMethod):
    """CRSAIL-style: query states far (in feature space) from already-collected demos."""

    name = "knn"

    def __init__(self, k: int = 5) -> None:
        self.k = k
        self._demo_feats: list[np.ndarray] = []

    def reset_round(self) -> None:  # demo memory persists across rounds by design
        pass

    def register_demo_state(self, feature: np.ndarray) -> None:
        self._demo_feats.append(np.asarray(feature, dtype=np.float64))

    def score(self, ctx: StepContext) -> float:
        if ctx.feature is None:
            raise ValueError("KNNCoverageQuery requires StepContext.feature")
        if not self._demo_feats:
            return 1.0  # nothing collected yet -> everything is maximally novel
        feats = np.stack(self._demo_feats)
        dists = np.linalg.norm(feats - np.asarray(ctx.feature, dtype=np.float64), axis=1)
        k = min(self.k, len(dists))
        return float(np.sort(dists)[:k].mean())


class HumanGatedQuery(QueryMethod):
    """Sim proxy for a human operator. A human intervenes when the policy is visibly
    about to do the wrong thing; we approximate that with privileged oracle-vs-policy
    action disagreement. This is the honest real-world competitor, not a strawman."""

    name = "human_gated"

    def score(self, ctx: StepContext) -> float:
        if ctx.oracle_disagreement is None:
            raise ValueError("HumanGatedQuery requires StepContext.oracle_disagreement")
        return float(ctx.oracle_disagreement)


class ConformalQuery(QueryMethod):
    """Our method: conformally-calibrated action uncertainty with IQT adaptation.

    For the budget-ranked sweep, `score` returns the threshold-normalized uncertainty
    (>1 == above the calibrated threshold). For the online-threshold variant the loop
    can instead call `uncertainty.is_uncertain`.
    """

    name = "conformal"

    def __init__(self, cfg: ConformalConfig) -> None:
        self.cfg = cfg
        self.uncertainty = ConformalActionUncertainty(
            target_alpha=cfg.target_alpha,
            gamma=cfg.iqt_gamma,
            calib_min=cfg.calib_min,
            use_iqt=cfg.use_iqt,
        )

    def reset_round(self) -> None:
        # Recalibration happens in the loop (it owns the held-out calibration rollouts);
        # here we just clear stale calibration so a round never reuses old thresholds.
        self.uncertainty.reset_calibration()

    def calibrate(self, calib_scores: list[float]) -> None:
        for s in calib_scores:
            self.uncertainty.add_calibration(s)

    def score(self, ctx: StepContext) -> float:
        raw = self.uncertainty.score(ctx.action_samples)
        # Before calibration, fall back to raw dispersion so round-1 ranking still works.
        return self.uncertainty.normalized(raw) if self.uncertainty.is_calibrated else raw

    def observe_episode(self, *, miscovered: bool) -> None:
        self.uncertainty.observe_label(miscovered)


class DispersionQuery(QueryMethod):
    """Action-space self-disagreement. Score = mean per-(step, dim) std across K
    sampled action chunks at each frame. Avoids the teacher-forced loss pathology
    by measuring uncertainty in the policy's output distribution directly.
    """

    name = "dispersion"

    def score(self, ctx: StepContext) -> float:
        return raw_dispersion(ctx.action_samples)


class DispersionQuotaQuery(QueryMethod):
    """Same score as `DispersionQuery`; the active loop applies a per-task quota
    at the selection step (round-robin under coverage). This class is a marker
    for the loop's dispatch — it carries no extra state.
    """

    name = "dispersion_quota"

    def score(self, ctx: StepContext) -> float:
        return raw_dispersion(ctx.action_samples)


def build_query_method(method: QueryMethodName, *, seed: int,
                       conformal: ConformalConfig) -> QueryMethod:
    """Factory used by the sweep runner."""
    if method == "random":
        return RandomQuery(seed=seed)
    if method == "entropy":
        return EntropyQuery()
    if method == "knn":
        return KNNCoverageQuery()
    if method == "human_gated":
        return HumanGatedQuery()
    if method == "conformal":
        return ConformalQuery(cfg=conformal)
    if method == "dispersion":
        return DispersionQuery()
    if method == "dispersion_quota":
        return DispersionQuotaQuery()
    raise ValueError(f"unknown query method: {method}")
