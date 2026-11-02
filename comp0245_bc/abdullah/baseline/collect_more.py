"""Record more expert runs with a hold after reaching, and arm-focused runs (round "more").

    python collect_more.py --start 300 --end 450 --out data/runs_more --workers 3                    # clean
    python collect_more.py --start 10150 --end 10225 --noise 0.05 0.1 0.15 --out data/runs_nudged_more --workers 3
    python collect_more.py --start 20120 --end 20180 --noise 0.05 0.1 --nudge-when reach --out data/runs_reach_more --workers 3
    python collect_more.py --start 30120 --end 30180 --kick 0.02 --out data/runs_kick_more --workers 3
    python collect_more.py --start 50000 --end 50200 --scene arm --out data/runs_arm --workers 3        # arm-focused

Same expert and file format as collect.py, with two additions:
    --hold S      keep recording S seconds after the hand first gets within 1 cm (default 3). The
                  template stops at that instant, so the original recordings have no rows inside 1 cm.
    --scene arm   start the base 20-50 cm straight behind its parking spot, facing it, so most of the run
                  is the arm reaching (about 3x more reach data per run). Half of the targets are at the
                  extreme heights where model K fails (0.15-0.30 m, 0.95-1.10 m), still inside the
                  training ranges. Starting straight behind the spot avoids the expert's orbit failure
                  (it only orbits when it starts near the spot at an angle).
Resumable: existing files are skipped. Seeds must not overlap other recordings or the benchmark.
"""
import argparse
import json
import os
import time
import traceback
from multiprocessing import Pool

import numpy as np

from quiet import run_main
from samplers import TRAIN_TARGET_Z, sample_target, training_scene

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)   # data/, splits/, benchmark/, configs/, models/ and RoboEnv/ are shared in the parent folder
EDGE_Z = ((0.15, 0.30), (0.95, 1.10))


def arm_scene(seed):
    """Target (half at extreme heights) and a start pose straight behind the expert's parking spot."""
    from expert_episode import STANDOFF_POINTS, STANDOFF_Z_POINTS
    rng = np.random.default_rng(seed)
    if rng.random() < 0.5:
        target = sample_target(rng, z_range=EDGE_Z[int(rng.integers(2))])
    else:
        target = sample_target(rng, z_range=TRAIN_TARGET_Z)
    b = np.arctan2(target[1], target[0])
    u = np.array([np.cos(b), np.sin(b)])
    standoff = float(np.interp(target[2], STANDOFF_Z_POINTS, STANDOFF_POINTS))
    goal = np.asarray(target[:2]) - standoff * u               # the expert's parking spot (template formula)
    start_xy = goal - rng.uniform(0.2, 0.5) * u                # straight behind it
    yaw = b + np.radians(rng.uniform(-5.0, 5.0))               # facing it, small heading error
    return target, np.array([start_xy[0], start_xy[1], yaw])


def _one(args):
    seed, out_dir, noise, nudge_when, kick, hold, scene = args
    path = os.path.join(out_dir, f"run_{seed:05d}.npz")
    if os.path.exists(path):
        return seed, "skipped", None
    try:
        from expert_episode import run_episode
        target, start = arm_scene(seed) if scene == "arm" else training_scene(seed)
        res = run_episode(target, start, seed, out_path=path, cmd_noise_std=noise, nudge_when=nudge_when,
                          kick_prob=kick, time_limit_s=900, hold_after_reach_s=hold)
        return seed, res["meta"]["outcome"], res["meta"]
    except Exception:
        with open(os.path.join(out_dir, "failures.log"), "a", encoding="utf-8") as f:
            f.write(f"--- seed {seed} ---\n{traceback.format_exc()}\n")
        return seed, "error", None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--end", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--noise", type=float, nargs="+", default=[0.0])
    ap.add_argument("--nudge-when", choices=["always", "reach"], default="always")
    ap.add_argument("--kick", type=float, default=0.0)
    ap.add_argument("--hold", type=float, default=3.0, help="seconds to keep recording after first reaching 1 cm")
    ap.add_argument("--scene", choices=["normal", "arm"], default="normal")
    a = ap.parse_args()

    out = os.path.abspath(a.out)
    for protected in ("runs", "runs_nudged", "runs_reach", "runs_kick"):
        if out == os.path.join(ROOT, "data", protected):
            raise SystemExit(f"refusing to add to the original recordings folder {a.out}; use a new folder")
    os.makedirs(out, exist_ok=True)
    jobs = [(s, out, a.noise[s % len(a.noise)], a.nudge_when, a.kick, a.hold, a.scene) for s in range(a.start, a.end)]
    print(f"Recording seeds {a.start}-{a.end - 1} ({a.scene}, hold {a.hold} s) into {a.out}", flush=True)
    counts, t0 = {}, time.time()
    with Pool(a.workers) as pool:
        for seed, outcome, m in pool.imap_unordered(_one, jobs):
            counts[outcome] = counts.get(outcome, 0) + 1
            extra = (f"final {m['final_ee_error'] * 100:4.1f} cm | reach {m['reach_time'] or 0:5.1f} s | "
                     f"sim {m['duration']:5.1f} s") if m else ""
            print(f"[{sum(counts.values()):4d}/{len(jobs)}] seed {seed}: {outcome:11s} {extra}", flush=True)
    print(f"Done in {(time.time() - t0) / 60:.1f} min: {json.dumps(counts)}")


if __name__ == "__main__":
    run_main(main, HERE)
