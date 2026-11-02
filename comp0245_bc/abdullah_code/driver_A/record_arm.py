"""Record the expert's arm from where the robot really ends up parked: the planner drives through obstacle scenes to a parking spot
moved at random from the ideal one, the expert's arm reaches from there and holds 5 s. Same file format as collect.py, so train.py can
add these with --extra-train. Why: model R's arm was trained parked almost exactly on the spot.

    python record_arm.py --out ../data/auto_arm --per-scenario 40 --workers 4
    python record_arm.py --out ../data/auto_arm_nudged --per-scenario 20 --nudge --workers 4
"""
import argparse
import json
import os
import traceback
from multiprocessing import Pool

import numpy as np

from nav import PlannerPolicy
from obstacle_scenes import SCENARIOS, SEED_BLOCK, make_scene
from quiet import run_main

HERE = os.path.dirname(os.path.abspath(__file__))
MAX_OFFSET_M = 0.30            # parking spot moved by up to this far, in any direction
MAX_HEADING_RAD = 0.30         # final heading off by up to this much (about 17 degrees)
FIRST_SEED = {False: 6_000, True: 7_000}
HOLD_S = 5.0


class ArmRecorder:
    def __init__(self, seed, nudge):
        self.policy, self.name, self.nudge, self.seed = PlannerPolicy(), "arm_teacher", nudge, seed

    def reset(self, world, target):
        p = self.policy
        p.reset(world, target)
        rng = self.rng = np.random.default_rng([self.seed, 21])
        a, r = rng.uniform(-np.pi, np.pi), rng.uniform(0, MAX_OFFSET_M)
        p.expert.goal_bearing = float(p.expert.goal_bearing + rng.uniform(-MAX_HEADING_RAD, MAX_HEADING_RAD))
        p.goal = p.expert.goal = p.expert.goal + r * np.array([np.cos(a), np.sin(a)])
        p.goal_centre = p._centre(p.goal, p.expert.goal_bearing)
        self.world, self.target, self.k, self.every, self.push, self.next_push = world, np.asarray(target, float), 0, max(1, int(round(0.01 / world.dt))), np.zeros(7), 0.0
        self.t_reached = None
        self.log = {k: [] for k in ("t", "base_pos", "base_quat", "base_bearing", "q_mes", "qd_mes", "ee_pos", "target", "torque_arm", "wheel_cmds",
                                    "tau_gravity", "nav_done", "holding")}

    def act(self, obs):
        import pinocchio as pin
        p = self.policy
        wheels, torque = p.act(obs)
        done = bool(p.expert.state.done)
        if self.t_reached is None and np.linalg.norm(self.target - obs["ee_pos"]) < 0.01:
            self.t_reached = obs["t"]
        if self.k % self.every == 0:
            w = self.world
            tau_g = pin.computeGeneralizedGravity(w.dyn_model.pin_model, w.dyn_model.pin_data, obs["q_pin"])[6:13]
            for key, v in (("t", obs["t"]), ("base_pos", obs["base_pos"]), ("base_quat", obs["base_quat"]), ("base_bearing", obs["base_bearing"]),
                           ("q_mes", obs["q_mes"]), ("qd_mes", obs["qd_mes"]), ("ee_pos", obs["ee_pos"]), ("target", self.target),
                           ("torque_arm", torque), ("wheel_cmds", wheels), ("tau_gravity", tau_g), ("nav_done", done), ("holding", self.t_reached is not None)):
                self.log[key].append(np.array(v, copy=True))
        self.k += 1
        if self.nudge and done:                 # label stays clean; the executed torque is pushed, as in the runs_reach recordings
            if obs["t"] >= self.next_push:
                self.push, self.next_push = self.rng.normal(0, 1.0, 7), obs["t"] + 0.2
            return wheels, torque + self.push
        return wheels, torque


def one(job):
    scene, out, nudge = job
    path = os.path.join(out, f"run_{scene['seed']:06d}.npz")
    if os.path.exists(path):
        return scene["seed"], "skipped"
    try:
        from rollout import rollout
        rec = ArmRecorder(scene["seed"] % 100_000, nudge)
        m, _ = rollout(rec, scene, max_time=120, hold_s=HOLD_S)
        # final_ee_error in cm -> m, as collect.py's meta. A run that reached but left the zone during a nudge still has good labels.
        outcome = "reached" if m.get("first_reach_s") is not None and m["outcome"] in ("reached", "reached_not_held") else m["outcome"]
        meta = {"seed": scene["seed"], "outcome": outcome, "final_ee_error": m["final_err_cm"] / 100, "scenario": scene["scenario"], "nudge": nudge}
        tmp = path + ".tmp.npz"
        np.savez_compressed(tmp, meta=np.array(json.dumps(meta)), **{k: np.asarray(v) for k, v in rec.log.items()})
        os.replace(tmp, path)
        return scene["seed"], outcome
    except Exception:
        with open(os.path.join(out, "failures.txt"), "a") as fh:
            fh.write(f"seed {scene['seed']}\n{traceback.format_exc()}\n")
        return scene["seed"], "error"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-scenario", type=int, default=40)
    ap.add_argument("--nudge", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    scenes = []
    for kind in SCENARIOS:
        seed, found = SEED_BLOCK["train"] + 10_000 * (kind - 10) + FIRST_SEED[a.nudge], 0
        while found < a.per_scenario:
            s = make_scene(kind, seed, found)
            seed += 1
            if s is not None:
                scenes.append(s)
                found += 1
    counts = {}
    with Pool(a.workers) as pool:
        for i, (seed, outcome) in enumerate(pool.imap_unordered(one, [(s, a.out, a.nudge) for s in scenes]), 1):
            counts[outcome] = counts.get(outcome, 0) + 1
            print(f"[{i:3d}/{len(scenes)}] seed {seed}: {outcome}", flush=True)
    print("done:", json.dumps(counts))


if __name__ == "__main__":
    run_main(main, HERE)
