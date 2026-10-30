"""Generate the benchmark scenes ONCE and freeze them in benchmark/scenes.json.

    python make_benchmark.py            # 50 scenes per tier, tiers 1-4

Run this before any training and never change the file afterwards: every model is tested on
exactly these scenes, and nobody can tune the test to suit a model.
"""
import argparse
import json
import os

from quiet import run_main
from samplers import benchmark_scene

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)   # data/, splits/, benchmark/, configs/, models/ and RoboEnv/ are shared in the parent folder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-tier", type=int, default=50)
    ap.add_argument("--out", default=os.path.join(ROOT, "benchmark", "scenes.json"))
    ap.add_argument("--force", action="store_true", help="overwrite an existing frozen file")
    a = ap.parse_args()

    if os.path.exists(a.out) and not a.force:
        print(f"{a.out} already exists and is frozen. Not overwriting (use --force only if you really mean it).")
        return
    scenes = [benchmark_scene(t, i) for t in (1, 2, 3, 4) for i in range(a.per_tier)]
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump({"description": "Frozen benchmark scenes. Do not edit.", "per_tier": a.per_tier,
                   "scenes": scenes}, fh, indent=1)
    print(f"Wrote {len(scenes)} scenes ({a.per_tier} per tier, tiers 1-4) to {a.out}")


if __name__ == "__main__":
    run_main(main, HERE)
