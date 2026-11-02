"""Record many expert runs: one .npz file per run.

Examples (from the RoboEnv pixi shell):
    python training/collect.py --start 0 --end 10                 # first 10 runs, to check things work
    python training/collect.py --start 0 --end 100 --workers 4    # person 1
    python training/collect.py --start 100 --end 200 --workers 4  # person 2, on another laptop
    python training/collect.py --start 10000 --end 10150 --noise 0.05 0.1 0.15 --out data/runs_nudged   # nudged runs
    python training/collect.py --start 20000 --end 20100 --noise 0.05 0.1 --nudge-when reach --out data/runs_reach
    python training/collect.py --start 30000 --end 30120 --kick 0.02 --out data/runs_kick      # velocity-recovery runs
    python training/collect.py --start 40000 --end 40160 --obs-noise 0 0.005 0.01 0.02 --delay 0 1 2 3 4 --out data/runs_disturbed

Safe to stop (Ctrl+C) and restart: runs whose file already exists are skipped.
Simulator warnings are hidden; set ROBOENV_VERBOSE=1 to see them.
"""
from _paths import ROOT, find_result  # noqa: F401  (project folder; lets the code folders import each other)
import argparse
import json
import os
import signal
import time
import traceback
from multiprocessing import Pool

from quiet import run_main
from samplers import training_scene

HERE = ROOT


def _ignore_ctrl_c():
    """Workers ignore Ctrl+C; the main process handles it and stops them cleanly."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _one(args):
    seed, out_dir, noise, time_limit, nudge_when, kick, obs_noise, delay_steps = args
    path = os.path.join(out_dir, f"run_{seed:05d}.npz")
    if os.path.exists(path):
        return seed, "skipped", None, None
    try:
        from expert_episode_noisy import run_episode  # import inside the worker (Windows-safe)
        target, start = training_scene(seed)
        res = run_episode(target, start, seed, out_path=path, cmd_noise_std=noise,
                          nudge_when=nudge_when, kick_prob=kick, obs_noise=obs_noise,
                          delay_steps=delay_steps, time_limit_s=time_limit)
        return seed, res["meta"]["outcome"], res["meta"], None
    except Exception as exc:
        with open(os.path.join(out_dir, "failures.log"), "a", encoding="utf-8") as f:
            f.write(f"--- seed {seed} ---\n{traceback.format_exc()}\n")
        return seed, "error", None, f"{type(exc).__name__}: {exc}"


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
                    help="sensor noise on the logged readings (tier 4 uses 0.005 0.01 0.02); several values = "
                         "each run gets one of them in turn (by seed)")
    ap.add_argument("--delay", type=int, nargs="+", default=[0],
                    help="command delay in 1 ms steps (tier 4 uses 1-4); several values = each run gets one "
                         "in turn (by seed). With 4 noise and 5 delay values every pairing appears once per 20 seeds")
    ap.add_argument("--time-limit", type=float, default=900, help="wall-clock seconds per run")
    a = ap.parse_args()

    if a.end <= a.start:
        print(f"Nothing to do: --end ({a.end}) must be larger than --start ({a.start}).")
        return

    os.makedirs(a.out, exist_ok=True)
    jobs = [(s, a.out, a.noise[s % len(a.noise)], a.time_limit, a.nudge_when, a.kick,
             a.obs_noise[s % len(a.obs_noise)], a.delay[s % len(a.delay)]) for s in range(a.start, a.end)]
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
