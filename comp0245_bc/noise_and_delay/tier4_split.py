"""Split tier 4 into its two disturbances, to find out which one breaks a model (Stage B, step 1).

    python noise_and_delay/tier4_split.py
    python evaluation/evaluate.py --policy trained/vK_kick.npz --tiers 4 --scenes benchmark/tier4_noise_only.json --out results_noise
    python evaluation/evaluate.py --policy trained/vK_kick.npz --tiers 4 --scenes benchmark/tier4_delay_only.json --out results_delay

Each tier-4 scene has sensor noise AND a command delay. This writes two copies of the same 50 scenes
(same targets, starts and seeds), one with only the noise and one with only the delay. The frozen
benchmark/scenes.json is only read, never changed.
"""
from _paths import ROOT, find_result  # noqa: F401  (project folder; lets the code folders import each other)
import copy
import json
import os

from quiet import run_main

HERE = ROOT
SRC = os.path.join(HERE, "benchmark", "scenes.json")


def main():
    with open(SRC) as fh:
        tier4 = [s for s in json.load(fh)["scenes"] if s["tier"] == 4]

    for name, keep in (("noise_only", "obs_noise"), ("delay_only", "delay_steps")):
        scenes = copy.deepcopy(tier4)
        for s in scenes:
            s["disturbance"] = {keep: s["disturbance"][keep]}
        out = os.path.join(HERE, "benchmark", f"tier4_{name}.json")
        with open(out, "w") as fh:
            json.dump({"description": f"Tier 4 with {keep} only (from scenes.json). Diagnostic, not the benchmark.",
                       "per_tier": len(scenes), "scenes": scenes}, fh, indent=1)
        print(f"Wrote {len(scenes)} tier-4 scenes with {keep} only to {out}")


if __name__ == "__main__":
    run_main(main, HERE)
