"""Split the recorded runs into train / validation / test BY RUN, and save the lists.

    python make_split.py                 # 70 / 15 / 15 of the runs that reached the target

Whole runs stay together: neighbouring timesteps are near-identical, so splitting by row would
leak training data into the test set and make scores look better than they are.
Everyone loads splits/split.json, so the whole team trains and tests on the same runs.
"""
import argparse
import glob
import json
import os

import numpy as np

from check_runs import load_run, _outcome
from quiet import run_main

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=[os.path.join(HERE, "data", "runs")],
                    help="one or more run folders; all their runs are split together")
    ap.add_argument("--out", default=os.path.join(HERE, "splits", "split.json"))
    ap.add_argument("--seed", type=int, default=0, help="shuffle seed (fixed, so the split is reproducible)")
    ap.add_argument("--fractions", type=float, nargs=3, default=[0.70, 0.15, 0.15])
    a = ap.parse_args()

    kept, excluded = [], []
    files = sorted(f for d in a.runs for f in glob.glob(os.path.join(d, "run_*.npz")))
    for f in files:
        try:
            meta, _ = load_run(f)
        except Exception:
            excluded.append({"file": os.path.basename(f), "reason": "unreadable"})
            continue
        if _outcome(meta) == "reached":
            kept.append(meta["seed"])
        else:
            excluded.append({"seed": meta["seed"], "reason": meta["outcome"]})

    if not kept:
        print(f"No usable runs in {', '.join(a.runs)}.")
        return

    rng = np.random.default_rng(a.seed)
    order = rng.permutation(sorted(kept)).tolist()
    n = len(order)
    n_train = int(round(a.fractions[0] * n))
    n_val = int(round(a.fractions[1] * n))
    split = {
        "train": sorted(order[:n_train]),
        "val": sorted(order[n_train:n_train + n_val]),
        "test": sorted(order[n_train + n_val:]),
        "excluded": excluded,
        "shuffle_seed": a.seed,
        "note": "Split by run. Test runs are for final offline scores only; never tune on them.",
    }
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(split, fh, indent=1)
    print(f"{n} usable runs -> train {len(split['train'])}, val {len(split['val'])}, test {len(split['test'])}; "
          f"excluded {len(excluded)}: {[e.get('seed', e.get('file')) for e in excluded]}")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    run_main(main, HERE)
