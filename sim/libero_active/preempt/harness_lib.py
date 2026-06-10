"""Core library for the E1 oracle-handover rollout harness.

Mirrors lerobot's eval pipeline (lerobot_eval.rollout / eval_main) but gives us
MANUAL chunk control so we can inject a mid-rollout disturbance and compare
override-vs-replan policies on a single, in-process (sync) LIBERO MuJoCo env.

Pipeline per control step (canonical, from lerobot_eval.rollout ~167-198):
  1. observation = preprocess_observation(observation)         (envs.preprocess_observation)
  2. observation["task"] = [task_description]
  3. observation = env_preprocessor(observation)               (LiberoProcessorStep)
  4. observation = preprocessor(observation)                   (policy normalizer/resizer)
  5. action = policy.select_action / predict_action_chunk      (we manage the chunk)
  6. action = postprocessor(action)
  7. action = env_postprocessor({ACTION: action})[ACTION]
  8. env.step(action.cpu().numpy())

We use predict_action_chunk(batch) -> (1, 50, 7) NORMALIZED chunk and execute
single actions from it ourselves, applying steps 6-7 to each (1,7) action.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from libero.libero import benchmark
from lerobot.configs.policies import PreTrainedConfig
from lerobot.envs import preprocess_observation
from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
from lerobot.envs.factory import make_env_pre_post_processors
from lerobot.envs.libero import LiberoEnv
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.utils.constants import ACTION
from lerobot.utils.random_utils import set_seed

# ---- fixed config (matches the working CLI; READ-ONLY checkpoint) ----
REPO_ROOT = Path(__file__).resolve().parents[1]
CKPT = str(
    REPO_ROOT / "results/fixedbuf_random_N20/seed0/train/checkpoints/004000/pretrained_model"
)
SUITE_NAME = "libero_spatial"
# CLI uses LiberoEnv defaults: obs 360x360, control_mode relative; policy resizes to 256.
OBS_HW = 360
DEVICE = "cuda"

# action layout: 6 pose dims (xyz + axis-angle) + 1 gripper
N_POSE_DIMS = 6
GRIPPER_DIM = 6


@dataclass
class Pipeline:
    policy: Any
    preprocessor: Any
    postprocessor: Any
    env_preprocessor: Any
    env_postprocessor: Any
    device: torch.device


def build_pipeline(ckpt: str = CKPT, device: str = DEVICE) -> Pipeline:
    """Build policy + all processors exactly like eval_main()."""
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # policy config from the checkpoint (chunk_size=50, n_action_steps=50, num_steps=10)
    policy_cfg = PreTrainedConfig.from_pretrained(ckpt)
    policy_cfg.pretrained_path = Path(ckpt)
    policy_cfg.device = device

    # env config: matches the working CLI (env.type=libero, env.task=libero_spatial)
    env_cfg = LiberoEnvConfig(task=SUITE_NAME)

    policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg)
    policy.eval()

    preprocessor_overrides = {
        "device_processor": {"device": str(policy.config.device)},
        "rename_observations_processor": {"rename_map": {}},
    }
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=policy_cfg.pretrained_path,
        preprocessor_overrides=preprocessor_overrides,
    )
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(
        env_cfg=env_cfg, policy_cfg=policy_cfg
    )

    return Pipeline(
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        env_preprocessor=env_preprocessor,
        env_postprocessor=env_postprocessor,
        device=get_safe_device(policy),
    )


def get_safe_device(policy: Any) -> torch.device:
    return next(iter(policy.parameters())).device


# ---- single in-process (sync) LIBERO env ----
_SUITE_CACHE: dict[str, Any] = {}


def _get_suite() -> Any:
    if SUITE_NAME not in _SUITE_CACHE:
        _SUITE_CACHE[SUITE_NAME] = benchmark.get_benchmark_dict()[SUITE_NAME]()
    return _SUITE_CACHE[SUITE_NAME]


def make_single_env(task_id: int, episode_index: int = 0) -> LiberoEnv:
    suite = _get_suite()
    return LiberoEnv(
        task_suite=suite,
        task_id=task_id,
        task_suite_name=SUITE_NAME,
        obs_type="pixels_agent_pos",
        observation_width=OBS_HW,
        observation_height=OBS_HW,
        init_states=True,
        episode_index=episode_index,
        control_mode="relative",
    )


def _batch_obs(obs: dict) -> dict:
    """Add a leading batch axis so it looks like a 1-env vec-env obs.

    A single LiberoEnv yields images of shape (H,W,C) and robot_state arrays of
    shape (D,). preprocess_observation auto-unsqueezes images (ndim==3), but the
    robot_state dict goes through _convert_nested_dict unchanged, and downstream
    LiberoProcessorStep._quat2axisangle REQUIRES shape (B,4). So we batch the
    robot_state arrays to (1, D) here, matching the SyncVectorEnv format.
    """
    out: dict = {}
    out["pixels"] = obs["pixels"]  # images: (H,W,C) -> preprocess unsqueezes to (1,H,W,C)
    rs = obs["robot_state"]
    batched_rs: dict = {}
    for group, sub in rs.items():
        batched_rs[group] = {}
        for k, v in sub.items():
            arr = np.asarray(v)
            batched_rs[group][k] = arr[None, ...]  # (1, D)
    out["robot_state"] = batched_rs
    return out


def build_policy_batch(pipe: Pipeline, raw_obs: dict, task_desc: str) -> dict:
    """Run canonical steps 1-4: env obs -> fully preprocessed policy batch."""
    observation = preprocess_observation(_batch_obs(raw_obs))  # step 1
    observation["task"] = [task_desc]  # step 2
    observation = pipe.env_preprocessor(observation)  # step 3
    observation = pipe.preprocessor(observation)  # step 4
    return observation


def execute_action(pipe: Pipeline, env: LiberoEnv, norm_action_1x7: torch.Tensor):
    """Take ONE normalized action (1,7), apply steps 6-7, env.step (1-D action).

    Returns (raw_obs, reward, terminated, info). Reads is_success from info
    BEFORE LiberoEnv.step auto-resets on terminated.
    """
    action = pipe.postprocessor(norm_action_1x7)  # step 6 (un-normalize)
    action = pipe.env_postprocessor({ACTION: action})[ACTION]  # step 7
    action_np = action.to("cpu").numpy()
    assert action_np.ndim == 2 and action_np.shape[0] == 1, f"bad action shape {action_np.shape}"
    raw_obs, reward, terminated, truncated, info = env.step(action_np[0])  # 1-D
    return raw_obs, float(reward), bool(terminated), info


# ---- plain chunked rollout (Stage 1): re-infer every n_action_steps ----
def plain_chunked_rollout(env: LiberoEnv, pipe: Pipeline, seed: int,
                          max_steps_cap: int | None = None) -> dict:
    """Re-infer a fresh chunk every n_action_steps; execute it open-loop in between.

    This reproduces the canonical select_action cadence but with explicit chunk
    management, so it should match the ~30% baseline.
    """
    set_seed(seed)
    pipe.policy.reset()
    raw_obs, info = env.reset(seed=seed)
    task_desc = env.task_description

    max_steps = env._max_episode_steps if max_steps_cap is None else min(max_steps_cap, env._max_episode_steps)
    n_action_steps = pipe.policy.config.n_action_steps

    chunk: torch.Tensor | None = None  # (1, T, 7) normalized
    chunk_pos = 0
    macro_calls = 0
    success = False

    for step in range(max_steps):
        if chunk is None or chunk_pos >= n_action_steps:
            batch = build_policy_batch(pipe, raw_obs, task_desc)
            with torch.inference_mode():
                chunk = pipe.policy.predict_action_chunk(batch)  # (1, T, 7)
            chunk_pos = 0
            macro_calls += 1

        a = chunk[:, chunk_pos, :]  # (1, 7) normalized
        chunk_pos += 1
        raw_obs, reward, terminated, info = execute_action(pipe, env, a)
        if bool(info.get("is_success", False)):
            success = True
        if terminated:
            return {"success": success, "n_steps": step + 1, "macro_calls": macro_calls,
                    "task": task_desc}

    return {"success": success, "n_steps": max_steps, "macro_calls": macro_calls, "task": task_desc}


# ---- Stage 2: unified rollout with perturbation + 5 variants ----
from preempt.perturb import perturb_nearest_object, perturb_object_by_name  # noqa: E402

VARIANTS = ("clean", "open_loop", "replan_only", "preempt_hold", "full_replan",
            "recovery_probe", "perturbed_start")


def _hold_action(last_chunk: torch.Tensor, chunk_pos: int) -> torch.Tensor:
    """HOLD = zeros on the 6 pose dims, keep last gripper command.

    Returns a (1,7) NORMALIZED action. The gripper command is taken from the
    action that was last issued from the stored chunk (index chunk_pos-1, clamped).
    """
    a = torch.zeros_like(last_chunk[:, 0, :])  # (1,7) zeros
    src = min(max(chunk_pos - 1, 0), last_chunk.shape[1] - 1)
    a[..., GRIPPER_DIM] = last_chunk[:, src, GRIPPER_DIM]
    return a


def _fresh_first_action(pipe: Pipeline, raw_obs: dict, task_desc: str) -> torch.Tensor:
    """First action of a fresh chunk computed from the current obs (1,7 normalized)."""
    batch = build_policy_batch(pipe, raw_obs, task_desc)
    with torch.inference_mode():
        chunk = pipe.policy.predict_action_chunk(batch)
    return chunk[:, 0, :]


def run_variant(env: LiberoEnv, pipe: Pipeline, seed: int, variant: str,
                perturb_step: int, latency: int, delta: float,
                measure_stale_harm: bool = False,
                stale_harm_horizon: int = 25,
                recovery_horizon: int = 8,
                start_perturb_object: str | None = None) -> dict:
    """Run one rollout of a given variant on a single env.

    All variants share IDENTICAL pre-perturbation behavior: set_seed(seed) is
    called first, the env init state is fixed by episode_index, and the base
    chunk cadence (re-infer every n_action_steps starting at step 0) is the same.
    """
    assert variant in VARIANTS, variant
    set_seed(seed)
    pipe.policy.reset()
    raw_obs, info = env.reset(seed=seed)
    task_desc = env.task_description

    # perturbed_start control: displace the (named) object at EPISODE START, then solve
    # from reset with normal chunked execution. Tests whether the disturbed object pose is
    # solvable in principle (isolates 'mid-trajectory recovery' from 'object-pose robustness').
    start_perturb_info: dict | None = None
    if variant == "perturbed_start" and start_perturb_object:
        start_perturb_info = perturb_object_by_name(env, start_perturb_object, delta=delta)
        raw_obs = _refresh_obs(env)

    max_steps = env._max_episode_steps
    n_action_steps = pipe.policy.config.n_action_steps
    eff_horizon = n_action_steps  # recovery_probe tightens this after t_p; others keep base cadence

    chunk: torch.Tensor | None = None  # current base chunk (1,T,7) normalized
    chunk_pos = 0
    macro_calls = 0
    success = False
    perturb_info: dict = {"perturbed": False}

    # replan/preempt windowing
    do_perturb = variant not in ("clean", "perturbed_start")
    stale_chunk: torch.Tensor | None = None  # snapshot at t_p (replan_only / preempt_hold)
    stale_pos_at_tp = 0
    window_end = -1  # = t_p + L while in the hold/stale window, else -1

    # stale-action-harm accumulation (secondary metric)
    harm_vals: list[float] = []

    for step in range(max_steps):
        # ---- inject disturbance exactly once, BEFORE selecting this step's action ----
        if do_perturb and step == perturb_step and not perturb_info["perturbed"]:
            perturb_info = perturb_nearest_object(env, delta=delta)
            # re-read obs so the post-perturbation state is visible to any replan
            raw_obs = _refresh_obs(env)
            if variant in ("replan_only", "preempt_hold"):
                stale_chunk = chunk  # snapshot the committed (now stale) chunk
                stale_pos_at_tp = chunk_pos
                window_end = step + latency
            elif variant == "recovery_probe":
                # best-case recovery: re-plan IMMEDIATELY from the perturbed obs (zero
                # latency) and then track with a tighter cadence. If this still fails, the
                # bottleneck is recovery capability, not stale-action execution timing.
                chunk_pos = n_action_steps  # forces a fresh replan in the base branch this step
                eff_horizon = recovery_horizon

        # ---- choose the action to execute this step ----
        in_window = window_end >= 0 and step < window_end

        if variant == "full_replan":
            # re-infer every step; execute first action of a fresh chunk
            a = _fresh_first_action(pipe, raw_obs, task_desc)
            macro_calls += 1

        elif in_window and variant == "preempt_hold":
            a = _hold_action(stale_chunk if stale_chunk is not None else chunk,
                             stale_pos_at_tp + (step - perturb_step))

        elif in_window and variant == "replan_only":
            # execute the STALE pre-perturbation chunk actions (no re-infer in window)
            idx = stale_pos_at_tp + (step - perturb_step)
            idx = min(idx, (stale_chunk.shape[1] - 1) if stale_chunk is not None else 0)
            a = stale_chunk[:, idx, :] if stale_chunk is not None else _zeros_action(pipe)

        else:
            # base cadence (clean / open_loop, replan/preempt after the window, recovery_probe)
            recompute = chunk is None or chunk_pos >= eff_horizon
            if window_end >= 0 and step == window_end:
                recompute = True  # the single replan at t_p + L
            if recompute:
                batch = build_policy_batch(pipe, raw_obs, task_desc)
                with torch.inference_mode():
                    chunk = pipe.policy.predict_action_chunk(batch)
                chunk_pos = 0
                macro_calls += 1
                if window_end >= 0 and step == window_end:
                    window_end = -1  # window closed; resume normal cadence
            a = chunk[:, chunk_pos, :]
            chunk_pos += 1

        # ---- secondary: stale-action harm over first H post-perturb steps ----
        if (measure_stale_harm and perturb_info["perturbed"]
                and perturb_step <= step < perturb_step + stale_harm_horizon):
            # save/restore RNG so this measurement does NOT advance the policy's
            # flow-sampling RNG and contaminate the primary rollout outcome.
            _rng_cpu = torch.get_rng_state()
            _rng_cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            fresh = _fresh_first_action(pipe, raw_obs, task_desc)
            torch.set_rng_state(_rng_cpu)
            if _rng_cuda is not None:
                torch.cuda.set_rng_state_all(_rng_cuda)
            harm_vals.append(float(torch.linalg.vector_norm(a - fresh).item()))

        # ---- execute ----
        raw_obs, reward, terminated, info = execute_action(pipe, env, a)
        if bool(info.get("is_success", False)):
            success = True
        if terminated:
            n_steps = step + 1
            break
    else:
        n_steps = max_steps

    out = {
        "variant": variant,
        "success": success,
        "n_steps": n_steps,
        "macro_calls": macro_calls,
        "task": task_desc,
        "perturb": start_perturb_info if variant == "perturbed_start" else perturb_info,
    }
    if measure_stale_harm:
        out["stale_harm"] = float(np.mean(harm_vals)) if harm_vals else None
        out["stale_harm_n"] = len(harm_vals)
    return out


def _refresh_obs(env: LiberoEnv) -> dict:
    """Re-read the env observation in the lerobot obs format (no env.step)."""
    rs = env._env.env
    return env._format_raw_obs(rs._get_observations())


def _zeros_action(pipe: Pipeline) -> torch.Tensor:
    return torch.zeros((1, 7), device=pipe.device)
