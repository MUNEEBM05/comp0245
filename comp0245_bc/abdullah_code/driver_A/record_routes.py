"""One drive per scenario on validation scenes, for the route figure: planner, learned driver, camera with no true position."""
import json
import os
from multiprocessing import Pool

import numpy as np

from quiet import run_main

HERE = os.path.dirname(os.path.abspath(__file__))
SETUPS = ["planner", "driverbrake:trained/driver_A_s1.npz", "cam"]


def one(job):
    scene, setup = job
    from nav import PlannerPolicy
    from rollout import make_policy, rollout
    from self_nav import SelfNavPolicy
    pol = SelfNavPolicy(PlannerPolicy(None), "cam", seed=scene["seed"] % 100_000) if setup == "cam" else make_policy(setup)
    m, traj = rollout(pol, scene, max_time=90, hold_s=5.0)
    xy = np.asarray(traj["base_pos"])[:, :2]
    return {"scenario": scene["scenario"], "setup": setup, "outcome": m["outcome"], "xy": np.round(xy[::10], 3).tolist()}


def main():
    with open(os.path.join(os.path.dirname(os.path.dirname(HERE)), "scenes_obstacles", "obstacles_val.json")) as fh:
        scenes = [s for s in json.load(fh)["scenes"] if s["index"] == 0]
    with Pool(2) as pool:
        rows = pool.map(one, [(s, u) for s in scenes for u in SETUPS], chunksize=1)
    with open(os.path.join(os.path.dirname(os.path.dirname(HERE)), "results", "2_obstacles", "abdullah_bases", "routes.json"), "w") as fh:
        json.dump({"scenes": scenes, "drives": rows}, fh)
    print({(r["scenario"], r["setup"]): r["outcome"] for r in rows})


if __name__ == "__main__":
    run_main(main, HERE)
