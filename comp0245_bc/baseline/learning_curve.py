"""Learning curve: does model K get better with more recorded runs?

    python learning_curve.py --fractions 0.25 0.5 0.75 --workers 3

For each fraction f, keeps a random f of the runs in EVERY recording set (clean training split,
nudged, reach-nudged, kicked), trains K's exact recipe on them, and drives the 50 tier-1 benchmark
scenes. Subsets are nested (the 25% runs are inside the 50% runs, ...) so the points differ only by
the added data. Validation is always the same 45 clean runs. The 100% point is K itself
(trained/vK_kick.npz, results50/vK_kick_tier1.csv).

Writes only new folders: data/lc/ (subset folders of links to the real run files), splits/lc/,
trained_lc/, results_lc/. Training needs PyTorch, so this calls RoboEnv's ml environment.
Resumable: a fraction whose model or results already exist is skipped.
"""
import argparse
import glob
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)   # data/, splits/, benchmark/, configs/, models/ and RoboEnv/ are shared in the parent folder
ML_PYTHON = os.path.join(ROOT, "RoboEnv", ".pixi", "envs", "ml", "bin", "python")
EXTRA = ["runs_nudged", "runs_reach", "runs_kick"]
K_RECIPE = ["--history", "deltas", "--goal", "--separate", "--reach-weight", "3", "--integral"]


def nested_pick(items, frac, seed):
    """The first frac of one fixed shuffle: smaller fractions are subsets of larger ones."""
    order = np.random.default_rng(seed).permutation(len(items))
    return [items[i] for i in sorted(order[:int(round(frac * len(items)))])]


def make_subset(frac, seed):
    tag = f"p{int(round(frac * 100)):03d}"
    with open(os.path.join(ROOT, "splits", "split.json")) as fh:
        split = json.load(fh)
    sub = dict(split, train=nested_pick(sorted(split["train"]), frac, seed))
    split_path = os.path.join(ROOT, "splits", "lc", f"split_{tag}.json")
    os.makedirs(os.path.dirname(split_path), exist_ok=True)
    with open(split_path, "w") as fh:
        json.dump(sub, fh)
    dirs = []
    for name in EXTRA:
        files = sorted(glob.glob(os.path.join(ROOT, "data", name, "run_*.npz")))
        d = os.path.join(ROOT, "data", "lc", tag, name)
        os.makedirs(d, exist_ok=True)
        for f in nested_pick(files, frac, seed):
            link = os.path.join(d, os.path.basename(f))
            if not os.path.exists(link):
                os.symlink(f, link)
        dirs.append(d)
    return tag, split_path, dirs, len(sub["train"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fractions", type=float, nargs="+", default=[0.25, 0.5, 0.75])
    ap.add_argument("--subset-seed", type=int, default=0, help="which random runs are kept")
    ap.add_argument("--train-seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()

    for frac in a.fractions:
        tag, split_path, dirs, n_clean = make_subset(frac, a.subset_seed)
        name = f"vK_lc_{tag}_s{a.train_seed}"
        model = os.path.join(HERE, "trained_lc", f"{name}.npz")
        print(f"\n=== {int(frac * 100)}%: {n_clean} clean training runs + "
              f"{', '.join(f'{len(os.listdir(d))} {os.path.basename(d)}' for d in dirs)} ===", flush=True)
        if not os.path.exists(model):
            subprocess.run([ML_PYTHON, os.path.join(HERE, "train_dagger.py"), "--name", name,
                            "--out-dir", os.path.join(HERE, "trained_lc"), "--split", split_path,
                            "--seed", str(a.train_seed), *K_RECIPE, "--extra-train", *dirs], check=True)
        if not os.path.exists(os.path.join(HERE, "results_lc", f"{name}_tier1.csv")):
            subprocess.run([sys.executable, os.path.join(HERE, "evaluate.py"), "--policy", model, "--tiers", "1",
                            "--n", "50", "--workers", str(a.workers), "--out", os.path.join(HERE, "results_lc")],
                           check=True)
    print("\nDone. Table: python compare.py --results results_lc  (K at 100%: results50/vK_kick_tier1.csv)")


if __name__ == "__main__":
    main()
