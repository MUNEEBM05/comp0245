"""Record the expert + navigation layer driving round obstacles, as training data for a LiDAR student.

The teacher is the template's expert with our navigation layer on top (NavPolicy: LiDAR + VFH + Bug 2):
it steers round the box / through the door, then hands back to the expert for parking and reaching.
Scenes are training scenes (seeds 90000+, never the benchmark), each with a box in the path (objects=3)
or a wall with a door (objects=4), the start moved 2.5 m from the parking spot like the benchmark tests.

Same file format as expert_episode.py (DATA_FORMAT.md) plus one extra array:
    lidar  (N, 24)  closest LiDAR hit in each 15 deg slice around the robot (features.lidar_sectors), 4 m = nothing
Labels are the commands the teacher actually sent (wheel_cmds, torque_arm). nav_done marks the rows where the
expert is reaching (the navigation layer has handed back and the expert's base phase is done).
Safe to stop and restart: runs whose file already exists are skipped.

    python navigation/record_nav.py --start 90000 --end 90240 --out data/runs_navobs --workers 2
"""
from _paths import ROOT, find_result  # noqa: F401  (project folder; lets the code folders import each other)
import argparse
import json
import os
import time
from multiprocessing import Pool

import numpy as np

HERE = ROOT


def _one(args):
    seed, out = args
    path = os.path.join(out, f"run_{seed:05d}.npz")
    if os.path.exists(path):
        return seed, "skip"
    os.environ["ROBOENV_HEADLESS"] = "1"
    import pinocchio as pin
    import rollout as R
    from features import lidar_sectors
    from samplers import training_scene, _parking_goal

    target, start = training_scene(seed)
    kind = 3 if (seed // 2) % 2 == 0 else 4            # door side alternates with seed parity, so mix both
    scene = {"target": list(map(float, target)), "start_pose": list(map(float, start)), "seed": seed,
             "disturbance": {"objects": kind}}
    expert = R.ExpertPolicy()
    nav = R.NavPolicy(expert)
    log = {k: [] for k in ("t", "base_pos", "base_quat", "base_bearing", "q_mes", "qd_mes", "ee_pos", "target",
                           "torque_arm", "wheel_cmds", "tau_gravity", "nav_done", "nav_active", "lidar")}
    st = {"k": 0, "lid": None}

    class Recorder:
        name = "expert_nav_rec"

        def reset(self, world, tgt):
            self.w = world
            nav.reset(world, tgt)

        def act(self, obs):
            wheels, torque = nav.act(obs)
            k = st["k"]
            if k % R.LIDAR_EVERY == 0:
                d, ang = R.lidar_scan(self.w, obs["base_pos"][:2])
                st["lid"] = lidar_sectors(d, ang, float(obs["base_bearing"]))
            if k % 10 == 0:                                 # log at 100 Hz like the expert recordings
                m = self.w.dyn_model
                g = pin.computeGeneralizedGravity(m.pin_model, m.pin_data, obs["q_pin"])[6:13]
                for key, v in (("t", obs["t"]), ("base_pos", obs["base_pos"]), ("base_quat", obs["base_quat"]),
                               ("base_bearing", obs["base_bearing"]), ("q_mes", obs["q_mes"]),
                               ("qd_mes", obs["qd_mes"]), ("ee_pos", obs["ee_pos"]), ("target", tgt_arr),
                               ("torque_arm", np.clip(torque, -R.MAX_TORQUE, R.MAX_TORQUE)),
                               ("wheel_cmds", np.clip(wheels, -R.MAX_WHEEL, R.MAX_WHEEL)), ("tau_gravity", g),
                               ("nav_done", bool(expert.state.done) and nav.mode == "learned"),
                               ("nav_active", nav.mode == "nav"), ("lidar", st["lid"])):
                    log[key].append(np.array(v, copy=True))
            st["k"] = k + 1
            return wheels, torque

    tgt_arr = np.asarray(target, float)
    t0 = time.time()
    m, _ = R.rollout(Recorder(), scene, hold_s=5)
    arrays = {k: np.asarray(v) for k, v in log.items()}
    arrays["nav_done"] = arrays["nav_done"].astype(bool)
    arrays["nav_active"] = arrays["nav_active"].astype(bool)
    meta = {"seed": seed, "target": list(map(float, target)), "start_pose": list(map(float, start)),
            "base_goal_pos": list(map(float, _parking_goal(tgt_arr))), "objects": kind,
            "outcome": m["outcome"], "final_ee_error": m.get("final_err_cm", np.nan) / 100.0,
            "teacher": "expert+nav", "n_logged": len(arrays["t"]), "wall_time_s": time.time() - t0,
            "nav_share": float(arrays["nav_active"].mean()) if len(arrays["t"]) else 0.0}
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, meta=json.dumps(meta), **arrays)
    os.replace(tmp, path)
    return seed, m["outcome"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--end", type=int, required=True)
    ap.add_argument("--out", default=os.path.join(HERE, "data", "runs_navobs"))
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    jobs = [(s, a.out) for s in range(a.start, a.end)]
    n_ok = 0
    with Pool(a.workers, maxtasksperchild=1) as pool:
        for i, (seed, outcome) in enumerate(pool.imap_unordered(_one, jobs), 1):
            n_ok += outcome == "reached"
            print(f"[{i}/{len(jobs)}] run {seed}: {outcome}  (reached so far: {n_ok})", flush=True)


if __name__ == "__main__":
    main()
