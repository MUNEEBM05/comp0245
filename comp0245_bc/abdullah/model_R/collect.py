"""Record many expert runs: one .npz file per run.

Examples (from the RoboEnv pixi shell):
    python collect.py --start 0 --end 10                 # first 10 runs, to check things work
    python collect.py --start 0 --end 100 --workers 4    # person 1
    python collect.py --start 100 --end 200 --workers 4  # person 2, on another laptop
    python collect.py --start 10000 --end 10150 --noise 0.05 0.1 0.15 --out data/runs_nudged   # nudged runs
    python collect.py --start 20000 --end 20100 --noise 0.05 0.1 --nudge-when reach --out data/runs_reach
    python collect.py --start 30000 --end 30120 --kick 0.02 --out data/runs_kick      # velocity-recovery runs
    python collect.py --start 50000 --end 50200 --obs-noise 0 0.005 0.01 0.02 --delay 0 1 \
                      --friction 0.6 1.2 --mass 0.8 1.2 --out data/runs_dr     # Stage B: domain randomisation
    python collect.py --start 60000 --end 60200 --hold 5 --out data/runs_hold          # arrive, then hold still 5 s
    python collect.py --start 61000 --end 61150 --hold 5 --z-bands 0.15 0.3 0.95 1.1 --out data/runs_extreme

Safe to stop (Ctrl+C) and restart: runs whose file already exists are skipped.
Simulator warnings are hidden; set ROBOENV_VERBOSE=1 to see them.
"""
import argparse
import json
import os
import signal
import time
import traceback
from multiprocessing import Pool

import numpy as np

from quiet import run_main
from samplers import training_scene

HERE = os.path.dirname(os.path.abspath(__file__))


def _ignore_ctrl_c():
    """Workers ignore Ctrl+C; the main process handles it and stops them cleanly."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _one(args):
    seed, out_dir, noise, time_limit, nudge_when, kick, dr, hold, z_bands, wide = args
    path = os.path.join(out_dir, f"run_{seed:05d}.npz")
    if os.path.exists(path):
        return seed, "skipped", None, None
    try:
        from expert_episode import run_episode  # import inside the worker (Windows-safe)
        if wide:      # objects anywhere (incl. unreachable): see samplers.WIDE_*
            from samplers import wide_training_scene
            target, start = wide_training_scene(seed)
        else:
            target, start = training_scene(seed)
        if z_bands:   # extra data at chosen heights (e.g. the extremes where the model fails most)
            rng = np.random.default_rng([seed, 3])
            lo, hi = z_bands[2 * int(rng.integers(len(z_bands) // 2)):][:2]
            target = np.array([target[0], target[1], rng.uniform(lo, hi)])
        res = run_episode(target, start, seed, out_path=path, cmd_noise_std=noise,
                          nudge_when=nudge_when, kick_prob=kick, time_limit_s=time_limit, hold_s=hold,
                          hold_unreachable=wide,
                          **randomise_world(seed, dr))
        return seed, res["meta"]["outcome"], res["meta"], None
    except Exception as exc:
        with open(os.path.join(out_dir, "failures.log"), "a", encoding="utf-8") as f:
            f.write(f"--- seed {seed} ---\n{traceback.format_exc()}\n")
        return seed, "error", None, f"{type(exc).__name__}: {exc}"


def randomise_world(seed, dr):
    """Stage B domain randomisation: this run's sensor noise, command delay, floor friction and base
    mass, drawn from the ranges given on the command line. Seeded by the run's seed, so every run can
    be rebuilt exactly. With no randomisation options this returns the template's world."""
    rng = np.random.default_rng([seed, 2])
    return {
        "obs_noise": float(rng.choice(dr["obs_noise"])),
        "delay_steps": int(rng.integers(dr["delay"][0], dr["delay"][1] + 1)),
        "floor_friction": float(rng.uniform(*dr["friction"])),
        "base_mass_scale": float(rng.uniform(*dr["mass"])),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", type=int, required=True, help="first seed (inclusive)")
    ap.add_argument("--end", type=int, required=True, help="last seed (exclusive)")
    ap.add_argument("--out", default=os.path.join(HERE, "data", "runs"), help="output folder")
    ap.add_argument("--workers", type=int, default=1, help="runs in parallel (about CPU cores - 1)")
    ap.add_argument("--noise", type=float, nargs="+", default=[0.0],
                    help="nudge strength as a fraction of each command's limit; several values = "
                         "each run gets one of them in turn (by seed)")
    ap.add_argument("--nudge-when", choices=["always", "reach"], default="always",
                    help="reach = nudge only while the arm is reaching (fine-correction data)")
    ap.add_argument("--kick", type=float, default=0.0,
                    help="per-10ms chance of a 50 ms torque kick while reaching (velocity-recovery data), e.g. 0.02")
    ap.add_argument("--obs-noise", type=float, nargs="+", default=[0.0],
                    help="sensor-noise levels; each run picks one at random (0 = clean)")
    ap.add_argument("--delay", type=int, nargs=2, default=[0, 0], metavar=("MIN", "MAX"),
                    help="command delay in ms, drawn per run between MIN and MAX")
    ap.add_argument("--friction", type=float, nargs=2, default=[1.0, 1.0], metavar=("LO", "HI"),
                    help="floor friction drawn per run (template: 1.0)")
    ap.add_argument("--mass", type=float, nargs=2, default=[1.0, 1.0], metavar=("LO", "HI"),
                    help="base mass multiplier drawn per run")
    ap.add_argument("--hold", type=float, default=0.0,
                    help="keep recording this many seconds after the expert arrives (hold-still data), e.g. 5")
    ap.add_argument("--z-bands", type=float, nargs="*", default=[],
                    help="target heights from these bands instead, as pairs LO HI ... e.g. 0.15 0.3 0.95 1.1")
    ap.add_argument("--wide", action="store_true",
                    help="objects anywhere: wide target range incl. unreachable ones; with --hold the expert also "
                         "keeps still for that long after giving up on an unreachable target")
    ap.add_argument("--time-limit", type=float, default=900, help="wall-clock seconds per run")
    a = ap.parse_args()

    if a.end <= a.start:
        print(f"Nothing to do: --end ({a.end}) must be larger than --start ({a.start}).")
        return

    os.makedirs(a.out, exist_ok=True)
    dr = {"obs_noise": a.obs_noise, "delay": a.delay, "friction": a.friction, "mass": a.mass}
    jobs = [(s, a.out, a.noise[s % len(a.noise)], a.time_limit, a.nudge_when, a.kick, dr, a.hold, a.z_bands, a.wide)
            for s in range(a.start, a.end)]
    counts, t0 = {}, time.time()
    print(f"Recording seeds {a.start}-{a.end - 1} into {a.out} with {a.workers} worker(s)")

    def report(result):
        seed, outcome, m, err = result
        counts[outcome] = counts.get(outcome, 0) + 1
        if m:
            extra = (f"final err {m['final_ee_error'] * 100:5.1f} cm | sim {m['duration']:5.1f} s | "
                     f"wall {m['wall_time_s']:5.1f} s")
        elif err:
            extra = f"{err} (details in failures.log)"
        else:
            extra = "already recorded"
        done = sum(counts.values())
        print(f"[{done:4d}/{len(jobs)}] seed {seed:5d}: {outcome:11s} {extra}", flush=True)

    if a.workers > 1:
        pool = Pool(a.workers, initializer=_ignore_ctrl_c)
        try:
            for r in pool.imap_unordered(_one, jobs):
                report(r)
            pool.close()
        except KeyboardInterrupt:
            pool.terminate()
            raise
        finally:
            pool.join()
    else:
        for j in jobs:
            report(_one(j))

    print(f"\nDone in {(time.time() - t0) / 60:.1f} min: {json.dumps(counts)}")
    if counts.get("error"):
        print(f"{counts['error']} run(s) hit an error; see {os.path.join(a.out, 'failures.log')}. "
              f"Run the same command again to retry them.")


if __name__ == "__main__":
    run_main(main, HERE)
