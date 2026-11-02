"""Fill the IK input cache before training, one thread per process.

Run with OMP_NUM_THREADS=1: with more threads on a busy machine the small matrix solves in run_ik
became about 50 times slower. Two trainers also computed the same runs at the same time.

    OMP_NUM_THREADS=1 python precompute_ik.py --shard 0 --shards 3 ../data/runs ../data/auto_arm
"""
import argparse
import glob
import os

from check_runs import load_run
from ik_features import run_ik


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    a = ap.parse_args()
    files = sorted(f for d in a.dirs for f in glob.glob(os.path.join(d, "run_*.npz")))[a.shard::a.shards]
    for i, f in enumerate(files):
        run_ik(load_run(f)[1])
        if i % 50 == 0:
            print(f"{i}/{len(files)}", flush=True)
    print("done", len(files), flush=True)


if __name__ == "__main__":
    main()
