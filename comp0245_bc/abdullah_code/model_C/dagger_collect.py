"""DAgger recordings: a trained model drives, the expert's command in every visited state is the label.

    python dagger_collect.py --policy trained/vK_kick.npz --start 40000 --end 40100 --workers 3 --out data/runs_dagger1
    python dagger_collect.py --policy trained_dagger/vK_dagger1.npz --start 41000 --end 41100 --workers 3 --out data/runs_dagger2

Behaviour cloning only ever sees states the expert visited, so when the model drifts somewhere new
(e.g. hand stalled 2 cm short of the target for 10 s) it has no example of what to do there.
DAgger (Ross, Gordon & Bagnell 2011): let the MODEL drive, ask the expert what it would have done
in each state the model reached, add those rows to training, retrain, repeat.

    --mode arm    the model drives the arm, the expert drives the wheels (default: the base already
                  parks well, so this targets the arm's failure states)
    --mode full   the model drives both
    --resync      keep the expert's arm reference within the lead it shows when IT is in control
                  (99th percentile per joint, hand within 5 cm: EXPERT_LEAD_P99 below). Without this
                  the reference winds up to its 0.15 rad cap while the model drives and the labels
                  become far stronger pushes than the expert ever uses (the runs_dagger1 failure).
    --scene edge  1/3 low targets (0.15-0.40 m), 1/3 far targets (1.8-2.0 m), 1/3 normal: where the
                  model hovers short of the target or hits its own base

The expert runs alongside the model on every step (its arm reference keeps updating as it would if
it were in control), so its command is a valid label for the state. Files have the same format as
expert_episode.py recordings (DATA_FORMAT.md): the label is torque_arm / wheel_cmds, what was
actually executed is torque_arm_sent / wheel_cmds_sent. Not recorded: qd_des_arm (only needed by
the --action reference ablation). Scenes come from samplers.training_scene, the same ranges as the
other recordings; use a seed block no other recording uses (clean 0+, nudged 10000+, reach 20000+,
kick 30000+). Resumable: existing files are skipped. Train on them with train_dagger.py.
"""
import argparse
import json
import os
import time
from multiprocessing import Pool

import numpy as np

from quiet import run_main

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))   # data/, splits/, benchmark/, configs/, models/ and RoboEnv/ are shared in the parent folder
# |q_des - q| per arm joint, 99th percentile over expert recordings (runs_more, runs_arm) with the
# hand within 5 cm of the target while reaching. Old DAgger runs sat at the 0.15 rad cap instead.
EXPERT_LEAD_P99 = np.array([0.005, 0.068, 0.15, 0.033, 0.013, 0.027, 0.005])


def edge_scene(seed):
    """Training-range scene biased to the hard cases: low targets, far targets, or normal."""
    from samplers import TRAIN_TARGET_R, sample_start, sample_target
    rng = np.random.default_rng(seed)
    kind = seed % 3
    if kind == 0:
        target = sample_target(rng, z_range=(0.15, 0.40))
    elif kind == 1:
        target = sample_target(rng, r_range=(1.8, TRAIN_TARGET_R[1]))
    else:
        target = sample_target(rng)
    return target, sample_start(rng)


class DaggerPolicy:
    """Executes the model's command, logs the expert's command for the same state at 100 Hz."""

    def __init__(self, model_path, mode, resync=False):
        from rollout import ExpertPolicy, MLPPolicy
        self.expert, self.model, self.mode, self.resync = ExpertPolicy(), MLPPolicy(model_path), mode, resync
        self.name = f"dagger_{self.model.name}_{mode}"

    def reset(self, world, target):
        self.w, self.target = world, np.asarray(target, float)
        self.expert.reset(world, target)
        self.model.reset(world, target)
        self.k = 0
        self.log_every = max(1, int(round(0.01 / world.dt)))
        self.log = {k: [] for k in [
            "t", "base_pos", "base_quat", "base_bearing", "q_mes", "qd_mes", "ee_pos", "target",
            "phase", "nav_done", "q_des_arm", "tau_gravity",
            "torque_arm", "wheel_cmds", "torque_arm_sent", "wheel_cmds_sent",
        ]}

    def act(self, obs):
        import pinocchio as pin
        from expert_episode import PHASE_CODES
        phase_before = "done" if self.expert.state.done else self.expert.state.phase
        if self.resync and self.expert.state.done:
            q = obs["q_mes"][4:11]
            self.expert.q_des = np.clip(self.expert.q_des, q - EXPERT_LEAD_P99, q + EXPERT_LEAD_P99)
        ew, et = self.expert.act(obs)
        mw, mt = self.model.act(obs)
        sent_w = ew if self.mode == "arm" else mw
        if self.k % self.log_every == 0:
            m = self.w.dyn_model
            L = self.log
            L["t"].append(obs["t"])
            L["base_pos"].append(obs["base_pos"])
            L["base_quat"].append(obs["base_quat"])
            L["base_bearing"].append(obs["base_bearing"])
            L["q_mes"].append(obs["q_mes"])
            L["qd_mes"].append(obs["qd_mes"])
            L["ee_pos"].append(obs["ee_pos"])
            L["target"].append(self.target)
            L["phase"].append(PHASE_CODES[phase_before])
            L["nav_done"].append(self.expert.state.done)
            L["q_des_arm"].append(self.expert.q_des.copy())
            L["tau_gravity"].append(pin.computeGeneralizedGravity(m.pin_model, m.pin_data, obs["q_pin"])[6:13])
            L["torque_arm"].append(et)
            L["wheel_cmds"].append(ew)
            L["torque_arm_sent"].append(mt)
            L["wheel_cmds_sent"].append(sent_w)
        self.k += 1
        return sent_w, mt


def record_one(args):
    seed, policy_path, mode, out_dir, max_time, resync, scene_kind = args
    path = os.path.join(out_dir, f"run_{seed:05d}.npz")
    if os.path.exists(path):
        return seed, "skipped", None, None
    try:
        from rollout import rollout
        from samplers import training_scene
        target, start = edge_scene(seed) if scene_kind == "edge" else training_scene(seed)
        scene = {"seed": seed, "target": list(map(float, target)), "start_pose": list(map(float, start))}
        pol = DaggerPolicy(policy_path, mode, resync)
        t0 = time.time()
        m, _ = rollout(pol, scene, max_time=max_time)
        arrays = {}
        for k, v in pol.log.items():
            a = np.asarray(v)
            arrays[k] = a if a.dtype == bool else a.astype(np.int8 if k == "phase" else np.float32)
        outcome = "reached" if m["success"] else m["outcome"]
        meta = {
            "seed": int(seed), "target": scene["target"], "start_pose": scene["start_pose"],
            "base_goal_pos": pol.expert.goal.tolist(), "base_goal_bearing": float(pol.expert.goal_bearing),
            "outcome": outcome, "final_ee_error": m["final_err_cm"] / 100, "min_ee_error": m["min_err_cm"] / 100,
            "duration": m["time_s"], "n_logged": int(len(arrays["t"])),
            "dagger_policy": os.path.basename(policy_path), "dagger_mode": mode, "max_time": max_time,
            "resync": resync, "scene": scene_kind,
            "wall_time_s": time.time() - t0,
        }
        os.makedirs(out_dir, exist_ok=True)
        tmp = path + ".tmp.npz"
        np.savez_compressed(tmp, meta=np.array(json.dumps(meta)), **arrays)
        os.replace(tmp, path)  # write-then-rename: a half-written file never looks finished
        return seed, outcome, m["min_err_cm"], m["final_err_cm"]
    except Exception as exc:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "failures.log"), "a", encoding="utf-8") as fh:
            fh.write(f"seed {seed}: {type(exc).__name__}: {exc}\n")
        return seed, "error", None, None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", required=True, help="trained model that drives (.npz)")
    ap.add_argument("--mode", choices=["arm", "full"], default="arm")
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--end", type=int, required=True)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--max-time", type=float, default=25.0,
                    help="seconds per drive; caps how many near-identical 'stuck' rows one run adds")
    ap.add_argument("--resync", action="store_true", help="cap the expert's reference lead (see above)")
    ap.add_argument("--scene", choices=["normal", "edge"], default="normal")
    ap.add_argument("--out", required=True, help="output folder, e.g. data/runs_dagger1")
    a = ap.parse_args()

    out = os.path.abspath(a.out)
    for protected in ("runs", "runs_nudged", "runs_reach", "runs_kick"):
        if out == os.path.join(ROOT, "data", protected):
            raise SystemExit(f"refusing to write DAgger runs into the expert recordings folder {a.out}")
    jobs = [(s, a.policy, a.mode, out, a.max_time, a.resync, a.scene) for s in range(a.start, a.end)]
    print(f"DAgger: {os.path.basename(a.policy)} drives ({a.mode}), {len(jobs)} scenes -> {a.out}")
    counts = {}
    with Pool(a.workers) as pool:
        for seed, outcome, mn, fin in pool.imap_unordered(record_one, jobs):
            counts[outcome] = counts.get(outcome, 0) + 1
            extra = f" closest {mn:.1f} cm, final {fin:.1f} cm" if mn is not None else ""
            print(f"  seed {seed}: {outcome}{extra}", flush=True)
    print("done: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))


if __name__ == "__main__":
    run_main(main, HERE)
