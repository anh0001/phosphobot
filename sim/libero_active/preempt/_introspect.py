"""De-risk: confirm how to reach LIBERO objects in the MuJoCo sim for perturbation."""
from __future__ import annotations
import numpy as np
from libero.libero import benchmark
from lerobot.envs.libero import LiberoEnv

suite = benchmark.get_benchmark_dict()["libero_spatial"]()
env = LiberoEnv(
    task_suite=suite, task_id=0, task_suite_name="libero_spatial",
    obs_type="pixels_agent_pos", observation_width=256, observation_height=256,
    init_states=True, episode_index=0, control_mode="relative",
)
obs, info = env.reset()
print("task:", env.task_description)
rs = env._env.env          # robosuite env
sim = rs.sim
raw = rs._get_observations()
eef = np.asarray(raw["robot0_eef_pos"])
print("eef_pos:", eef)

# enumerate free joints (mjtJoint.mjJNT_FREE == 0) -> movable objects
free = []
for jid in range(sim.model.njnt):
    if int(sim.model.jnt_type[jid]) == 0:  # FREE
        name = sim.model.joint_id2name(jid)
        adr = int(sim.model.jnt_qposadr[jid])
        pos = np.asarray(sim.data.qpos[adr:adr + 3])
        free.append((name, adr, pos))
print(f"\n{len(free)} free-joint (movable) bodies:")
for name, adr, pos in free:
    print(f"  {name:40s} qposadr={adr:4d} xy=({pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f}) d_eef={np.linalg.norm(pos[:2]-eef[:2]):.3f}")

# nearest movable object to eef in XY
if free:
    nearest = min(free, key=lambda t: np.linalg.norm(t[2][:2] - eef[:2]))
    name, adr, pos = nearest
    print(f"\nnearest movable -> {name} at {pos}")
    # nudge laterally +4cm in x, settle, confirm it moved & no crash
    before = np.asarray(sim.data.qpos[adr:adr + 3]).copy()
    sim.data.qpos[adr] += 0.04
    sim.forward()
    after = np.asarray(sim.data.qpos[adr:adr + 3]).copy()
    print(f"nudge ok: before={before} after={after}")

print("\ncheck_success ->", rs.check_success())
print("INTROSPECT_OK")
env.close()
