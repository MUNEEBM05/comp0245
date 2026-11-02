"""Bayesian optimisation of model R's training settings and sensor filter (as in Lab 3: skopt gp_minimize).

    python bayes_opt.py --calls 12            # resumable: re-run the same command to continue

What is tuned (the "black box" = train a model, then let it DRIVE):
    learning rate (log 3e-4 .. 3e-3), batch size, network width and depth, reach weight, filter window (ms)
What is scored: share of TUNING scenes where the hand reaches AND holds within 1 cm for 5 s, averaged over
    the clean scenes and the same scenes with sensor noise (0.01), with the position filter switched on.
    The score is negated because gp_minimize minimises.
Fair setup: 20 separate tuning scenes (benchmark/scenes_bo.json, tier-1 recipe, indices 200-219), never
    the benchmark scenes 0-49 or the fresh test scenes 50-99. Each trial is trained with the same data and
    seed as model R, so only the settings differ.
Surrogate: Gaussian process; acquisition: expected improvement (EI). The first 4 trials are random.
Every finished trial is saved to bo/trials.json, so a stopped run carries on where it left off.
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
OUT = os.path.join(HERE, "bo")
DATA = ["data/runs_nudged", "data/runs_reach", "data/runs_kick", "data/runs_hold", "data/runs_extreme"]


def tuning_scenes():
    path = os.path.join(HERE, "benchmark", "scenes_bo.json")
    if not os.path.exists(path):
        from samplers import benchmark_scene
        scenes = [benchmark_scene(1, i) for i in range(200, 220)]
        with open(path, "w") as fh:
            json.dump({"description": "Bayesian-optimisation tuning scenes (tier-1 recipe, indices 200-219). "
                       "Never used for the benchmark or the final test.", "per_tier": 20, "scenes": scenes}, fh, indent=1)
    return path


def held_share(policy, scenes, extra, name):
    """Run evaluate.py; return the share of scenes reached and held 5 s."""
    out = os.path.join(OUT, "results")
    cmd = [PY, "evaluate.py", "--policy", policy, "--tiers", "1", "--workers", "2", "--scenes", scenes,
           "--out", out, "--plots", "0", "--hold", "5"] + extra
    subprocess.run(cmd, cwd=HERE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True,
                   env=dict(os.environ, ROBOENV_HEADLESS="1"))
    import csv
    files = [f for f in os.listdir(out) if f.startswith(name) and f.endswith("_tier1.csv")]
    f = max((os.path.join(out, f) for f in files), key=os.path.getmtime)
    rows = list(csv.DictReader(open(f)))
    return sum(r["success"] == "True" for r in rows) / len(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calls", type=int, default=12, help="total number of trials")
    a = ap.parse_args()
    from skopt import gp_minimize
    from skopt.space import Categorical, Integer, Real

    os.makedirs(OUT, exist_ok=True)
    scenes = tuning_scenes()
    space = [Real(3e-4, 3e-3, prior="log-uniform", name="lr"), Categorical([512, 1024, 2048], name="batch"),
             Categorical([128, 192, 256, 384], name="width"), Integer(2, 4, name="depth"),
             Real(1.0, 5.0, name="reach_weight"), Integer(5, 30, name="filter_ms")]
    log_path = os.path.join(OUT, "trials.json")
    trials = json.load(open(log_path)) if os.path.exists(log_path) else []

    def objective(x):
        lr, batch, width, depth, rw, filt = x
        k = len(trials)
        name = f"bo_t{k:02d}"
        t0 = time.time()
        model = os.path.join(HERE, "trained", f"{name}.npz")
        if not os.path.exists(model):
            subprocess.run([PY, "train.py", "--name", name, "--ik", "--history", "deltas", "--goal", "--separate",
                            "--integral", "--reach-weight", f"{rw}", "--lr", f"{lr}", "--batch", f"{batch}",
                            "--hidden", *[str(width)] * depth, "--extra-train", *DATA],
                           cwd=HERE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        pol = f"trained/{name}.npz:filt{filt}noqd"
        clean = held_share(pol, scenes, [], name)
        noisy = held_share(pol, scenes, ["--set", "obs_noise=0.01"], name + "_filt" + str(filt) + "noqd_obs_noise")
        score = (clean + noisy) / 2
        trials.append({"trial": k, "lr": lr, "batch": int(batch), "width": int(width), "depth": int(depth),
                       "reach_weight": rw, "filter_ms": int(filt), "held_clean": clean, "held_noisy": noisy,
                       "score": score, "minutes": round((time.time() - t0) / 60, 1)})
        json.dump(trials, open(log_path + ".tmp", "w"), indent=1)
        os.replace(log_path + ".tmp", log_path)
        print(f"trial {k:2d}: lr {lr:.2e} batch {batch} {depth}x{width} reach {rw:.1f} filter {filt} ms "
              f"-> held clean {clean:.0%}, noisy {noisy:.0%}, score {score:.2f} ({trials[-1]['minutes']} min)", flush=True)
        return -score

    left = a.calls - len(trials)
    if left > 0:
        x0 = [[t["lr"], t["batch"], t["width"], t["depth"], t["reach_weight"], t["filter_ms"]] for t in trials] or None
        y0 = [-t["score"] for t in trials] or None
        # model R's own settings (+ 10 ms filter) as the first trial, so we know the starting point
        if not trials:
            x0, y0 = [[1e-3, 1024, 256, 3, 3.0, 10]], None
        gp_minimize(objective, space, n_calls=left, n_initial_points=max(0, 4 - len(trials)), x0=x0, y0=y0,
                    acq_func="EI", random_state=len(trials))
    best = max(trials, key=lambda t: t["score"])
    print("best trial:", json.dumps(best, indent=1))
    json.dump(best, open(os.path.join(OUT, "best.json"), "w"), indent=1)


if __name__ == "__main__":
    from quiet import run_main
    run_main(main, HERE)
