"""Test the no-true-position robot (self_nav.py) on obstacle scenes. Reached and held 5 s within 1 cm, scored with the true pose.

    python eval_selfnav.py --mode cam --drive planner --scenes scenes/obstacles_val.json --n 10 --workers 4 --out results/stage6
    python eval_selfnav.py --mode search --drive trained/driver_A_s1.npz --arm trained/vR_ik.npz ...
"""
import argparse
import csv
import json
import os
from multiprocessing import Pool

import numpy as np

from quiet import run_main

HERE = os.path.dirname(os.path.abspath(__file__))


def one(job):
    scene, mode, drive, arm, max_time, freeze, brake = job
    from driver import DriverPolicy
    from nav import PlannerPolicy
    from rollout import rollout
    import self_nav
    from self_nav import SelfNavPolicy
    if freeze is not None:
        self_nav.FREEZE_HAND_M = freeze
    inner = PlannerPolicy(arm) if drive == "planner" else DriverPolicy(drive, arm, brake=brake)
    pol = SelfNavPolicy(inner, mode, seed=scene["seed"] % 100_000)
    m, _ = rollout(pol, scene, max_time=max_time, hold_s=5.0)
    return {"scenario": scene.get("scenario", f"tier{scene['tier']}"), "index": scene["index"], "outcome": m["outcome"], "first_reach_s": m.get("first_reach_s"),
            "final_err_cm": m.get("final_err_cm"), "min_err_cm": m.get("min_err_cm"), "time_s": m["time_s"], "path_m": m["base_path_m"], **pol.stats}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["odo", "cam", "search"], required=True)
    ap.add_argument("--drive", default="planner")
    ap.add_argument("--arm", default=None)
    ap.add_argument("--scenes", default=os.path.join(HERE, "scenes", "obstacles_val.json"))
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--tiers", type=int, nargs="+", default=None, help="only these tiers (for the team's benchmark)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-time", type=float, default=110.0)
    ap.add_argument("--brake", action="store_true", help="learned drivers: stop the wheels once parked")
    ap.add_argument("--freeze", type=float, default=None, help="stop looking when the hand is this close to the target, m")
    ap.add_argument("--out", default=os.path.join(HERE, "results", "stage6"))
    a = ap.parse_args()
    with open(a.scenes) as fh:
        scenes = [s for s in json.load(fh)["scenes"] if s["index"] < a.n and (a.tiers is None or s["tier"] in a.tiers)]
    with Pool(a.workers) as pool:
        rows = pool.map(one, [(s, a.mode, a.drive, a.arm, a.max_time, a.freeze, a.brake) for s in scenes], chunksize=1)
    d = "planner" if a.drive == "planner" else os.path.splitext(os.path.basename(a.drive))[0]
    arm = os.path.splitext(os.path.basename(a.arm))[0] if a.arm else "expert"
    tag = next((t for t, k in (("test", "test"), ("val", "val"), ("bench", "scenes.json")) if k in os.path.basename(a.scenes)), "val")
    name = f"{a.mode}_{d}_{arm}_{tag}"
    os.makedirs(a.out, exist_ok=True)
    keys = sorted({k for r in rows for k in r})
    with open(os.path.join(a.out, name + ".csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    def med(k):
        v = [r[k] for r in rows if r[k] is not None]
        return float(np.median(v)) * 100 if v else float("nan")
    print(name)
    for sc in sorted({r["scenario"] for r in rows}) + ["ALL"]:
        R = [r for r in rows if sc == "ALL" or r["scenario"] == sc]
        out = {}
        for r in R:
            out[r["outcome"]] = out.get(r["outcome"], 0) + 1
        print(f"  {sc:10s} n={len(R):3d} reached {sum(r['first_reach_s'] is not None for r in R):3d} held {sum(r['outcome'] == 'reached' for r in R):3d} {out}")
    print(f"  drift at park {med('drift_at_park_m'):.1f} cm | target error before camera {med('park_target_err_before_m'):.1f} cm, after {med('park_target_err_m'):.1f} cm | "
          f"camera saw target at park {sum(r['seen_at_park'] is True for r in rows)}/{len(rows)} | found on start spin {sum(r['found_after_s'] is not None for r in rows)}/{len(rows)}")


if __name__ == "__main__":
    run_main(main, HERE)
