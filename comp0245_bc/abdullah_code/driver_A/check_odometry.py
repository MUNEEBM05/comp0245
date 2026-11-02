"""How far off are the encoders and gyro by the time the robot parks? The planner drives using the true pose; the estimates run
alongside and change nothing. Validation scenes only.

    python check_odometry.py --n 6 --workers 3
"""
import argparse
import json
import os
from multiprocessing import Pool

import numpy as np

from localisation import CentreOdometry, Odometry, WheelsOnly
from quiet import run_main
from robotnik_arm_control import wrap_angle
from sensors import Accelerometer, Gyro, WheelEncoders

HERE = os.path.dirname(os.path.abspath(__file__))
START_STILL_S = 2.0


class Probe:
    def __init__(self, policy):
        self.policy, self.name = policy, policy.name

    def reset(self, world, target):
        self.policy.reset(world, target)
        self.enc, self.gyro, self.acc = WheelEncoders(seed=int(self.seed)), Gyro(world, seed=int(self.seed)), Accelerometer(world, seed=int(self.seed))
        self.dt, self.k, self.odo, self.wheels, self.at_park = 0.01, 0, None, None, None
        self.every = max(1, int(round(0.01 / world.dt)))
        self.wrong_v = 0.0

    def act(self, obs):
        out = self.policy.act(obs)
        if self.k % self.every == 0:
            xy, yaw = obs["base_pos"][:2], float(obs["base_bearing"])
            if self.odo is None:
                self.odo, self.wheels = Odometry(xy[0], xy[1], yaw), WheelsOnly(xy[0], xy[1], yaw)
                self.centre, self.accel = CentreOdometry(xy[0], xy[1], yaw), CentreOdometry(xy[0], xy[1], yaw, use_accel=True)
            wheels = self.enc.read(obs["qd_mes"])
            g, acc = self.gyro.read(self.dt), self.acc.read(self.dt)
            est, est_w = self.odo.update(wheels, g, self.dt), self.wheels.update(wheels, self.dt)
            est_c, est_a = self.centre.update(wheels, g, self.dt), self.accel.update(wheels, g, self.dt, acc)
            if self.at_park is None and self.policy.expert.state.done:
                self.at_park = (float(np.linalg.norm(est[:2] - xy)), abs(float(wrap_angle(est[2] - yaw))),
                                float(np.linalg.norm(est_w[:2] - xy)), abs(float(wrap_angle(est_w[2] - yaw))), obs["t"],
                                float(np.linalg.norm(est_c[:2] - xy)), float(np.linalg.norm(est_a[:2] - xy)))
        self.k += 1
        return out


def one(scene):
    from nav import PlannerPolicy
    from rollout import rollout
    pr = Probe(PlannerPolicy(start_still_s=START_STILL_S))
    pr.seed = scene["seed"] % 100000
    m, _ = rollout(pr, scene, max_time=90)
    return scene["scenario"], scene["index"], m["outcome"], m["base_path_m"], pr.at_park


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    with open(os.path.join(os.path.dirname(os.path.dirname(HERE)), "scenes_obstacles", "obstacles_val.json")) as fh:
        scenes = [s for s in json.load(fh)["scenes"] if s["index"] < a.n]
    with Pool(a.workers) as pool:
        rows = pool.map(one, scenes)
    done = [r for r in rows if r[4] is not None]
    print(f"{len(done)} of {len(rows)} drives parked")
    for name in ("clear", "sparse", "cluttered", "gap", "blocked"):
        R = [r for r in done if r[0] == name]
        print(f"  {name:10s} path {np.mean([r[3] for r in R]):4.1f} m | encoders + gyro: {100 * np.median([r[4][0] for r in R]):5.1f} cm, "
              f"{np.degrees(np.median([r[4][1] for r in R])):4.1f} deg | wheels only: {100 * np.median([r[4][2] for r in R]):5.1f} cm, "
              f"{np.degrees(np.median([r[4][3] for r in R])):4.1f} deg")
    for label, i in (("rotation-centre", 5), ("rotation-centre + accelerometer", 6)):
        v = np.array([r[4][i] for r in done]) * 100
        print(f"  {label}: median {np.median(v):.1f} cm, worst {v.max():.1f} cm")
    e = np.array([r[4][0] for r in done]) * 100
    print(f"  all: encoders + gyro position error median {np.median(e):.1f} cm, worst {e.max():.1f} cm, {100 * np.median(e / (np.array([r[3] for r in done]) * 100)):.1f}% of distance driven")


if __name__ == "__main__":
    run_main(main, HERE)
