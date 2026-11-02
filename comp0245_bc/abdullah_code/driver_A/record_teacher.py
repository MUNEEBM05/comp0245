"""Record the teacher: the classical planner + expert arm drives through obstacle scenes, and every 10 ms we save what the sensors
read (inputs) and what the planner commanded (labels). One .npz per drive, in a shared data/ folder. Resumable.

    python record_teacher.py --out ../data/auto_teacher --per-scenario 60 --workers 3                       # clean drives
    python record_teacher.py --out ../data/auto_teacher_nudged --per-scenario 20 --nudge --workers 3        # pushed off the route
    python record_teacher.py --out ../data/auto_teacher_val --per-scenario 8 --block val --workers 3        # validation

--nudge: every 0.5 s a random extra left/right wheel speed is executed, but the saved label stays the planner's clean command, so
the data shows how to get back on the route (the same idea as the earlier nudged recordings).
True pose is saved for building inputs now and for scoring the sensor estimates later (step 6).
"""
import argparse
import json
import os
import traceback
from multiprocessing import Pool

import numpy as np

from driver_features import SECTORS, lidar_sectors
from nav import PlannerPolicy
from obstacle_scenes import SCENARIOS, SEED_BLOCK, make_scene
from quiet import run_main
from sensors import Gyro, WheelEncoders

HERE = os.path.dirname(os.path.abspath(__file__))
NUDGE_RAD_S = 0.7
NUDGE_EVERY_S = 0.5
NUDGE_SEED_OFFSET = 5_000      # nudged drives use later seeds in the same scenario's block, so no scene is repeated


class Recorder:
    def __init__(self, seed, nudge=False):
        self.policy, self.name, self.nudge, self.seed = PlannerPolicy(), "teacher", nudge, seed

    def reset(self, world, target):
        self.policy.reset(world, target)
        self.enc, self.gyro = WheelEncoders(seed=self.seed), Gyro(world, seed=self.seed)
        self.rng = np.random.default_rng([self.seed, 11])
        self.k, self.every, self.offset, self.next_nudge = 0, max(1, int(round(0.01 / world.dt))), np.zeros(4), 0.0
        self.log = {k: [] for k in ("t", "xy", "yaw", "centre", "lidar", "waypoint", "wp_last", "wheel_cmds", "encoders", "gyro",
                                    "goal_centre", "goal_bearing", "navigating", "done")}

    def act(self, obs):
        p = self.policy
        wheels, torque = p.act(obs)
        if self.k % self.every == 0:
            xy, yaw = obs["base_pos"][:2].copy(), float(obs["base_bearing"])
            centre = p._centre(xy, yaw)
            path = p.path if p.path is not None else [centre, p.goal_centre]
            i = min(p.follower.next, len(path) - 1) if p.navigating else len(path) - 1
            L = self.log
            for key, value in (("t", obs["t"]), ("xy", xy), ("yaw", yaw), ("centre", centre), ("lidar", lidar_sectors(p.last_ranges)),
                               ("waypoint", path[i]), ("wp_last", float(i == len(path) - 1)), ("wheel_cmds", wheels),
                               ("encoders", self.enc.read(obs["qd_mes"])), ("gyro", self.gyro.read(0.01)), ("goal_centre", p.goal_centre),
                               ("goal_bearing", p.expert.goal_bearing), ("navigating", p.navigating), ("done", p.expert.state.done)):
                L[key].append(value)
        self.k += 1
        if self.nudge and p.navigating:
            if obs["t"] >= self.next_nudge:
                d = float(self.rng.normal(0, NUDGE_RAD_S))
                self.offset, self.next_nudge = np.array([d, -d, -d, d]), obs["t"] + NUDGE_EVERY_S
            return wheels + self.offset, torque
        return wheels, torque


def one(job):
    scene, out, nudge = job
    path = os.path.join(out, f"run_{scene['seed']:06d}.npz")
    if os.path.exists(path):
        return scene["seed"], "skipped", 0.0
    try:
        from rollout import rollout
        rec = Recorder(scene["seed"] % 100_000, nudge)
        m, _ = rollout(rec, scene, max_time=120)
        meta = {"seed": scene["seed"], "scenario": scene["scenario"], "outcome": m["outcome"], "time_s": m["time_s"], "nudge": nudge,
                "base_path_m": m["base_path_m"], "shortest_path_m": scene["shortest_path_m"], "target": scene["target"],
                "start_pose": scene["start_pose"], "obstacles": scene["obstacles"]}
        tmp = path + ".tmp.npz"
        np.savez_compressed(tmp, meta=np.array(json.dumps(meta)), **{k: np.asarray(v, np.float32) for k, v in rec.log.items()})
        os.replace(tmp, path)
        return scene["seed"], m["outcome"], m["time_s"]
    except Exception:
        with open(os.path.join(out, "failures.txt"), "a") as fh:
            fh.write(f"seed {scene['seed']}\n{traceback.format_exc()}\n")
        return scene["seed"], "error", 0.0


def scenes_for(block, per_scenario, nudge):
    out = []
    for kind in SCENARIOS:
        first = SEED_BLOCK[block] + 10_000 * (kind - 10) + (NUDGE_SEED_OFFSET if nudge else 0)
        seed, found = first, 0
        while found < per_scenario:
            if seed >= SEED_BLOCK[block] + 10_000 * (kind - 9):
                raise RuntimeError(f"{SCENARIOS[kind]}: too few valid layouts in its seed block")
            s = make_scene(kind, seed, found)
            seed += 1
            if s is not None:
                out.append(s)
                found += 1
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-scenario", type=int, default=60)
    ap.add_argument("--block", choices=["train", "val"], default="train")
    ap.add_argument("--nudge", action="store_true")
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    scenes = scenes_for(a.block, a.per_scenario, a.nudge)
    counts = {}
    with Pool(a.workers) as pool:
        for i, (seed, outcome, t) in enumerate(pool.imap_unordered(one, [(s, a.out, a.nudge) for s in scenes]), 1):
            counts[outcome] = counts.get(outcome, 0) + 1
            print(f"[{i:3d}/{len(scenes)}] seed {seed}: {outcome} {t:.1f} s", flush=True)
    print("done:", json.dumps(counts))


if __name__ == "__main__":
    run_main(main, HERE)
