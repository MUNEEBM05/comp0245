"""Watch one benchmark scene drive in the GUI (for inspecting where a model fails).

    python watch.py --policy trained/vK_kick.npz --tier 1 --index 40   # the crash
    python watch.py --policy trained/vK_kick.npz --tier 1 --index 18   # a near-miss timeout
    python watch.py --policy expert              --tier 1 --index 40   # the expert on the same scene

Scene seed/target/start come straight from benchmark/scenes.json, so this is the exact
scene evaluate.py scored. Prints the end-of-drive metrics after the window closes.
"""
import argparse
import json
import os

from quiet import run_main
from rollout import make_policy, rollout

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)   # data/, splits/, benchmark/, configs/, models/ and RoboEnv/ are shared in the parent folder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True, help="'expert' or path to a trained .npz / .pkl")
    ap.add_argument("--tier", type=int, default=1)
    ap.add_argument("--index", type=int, required=True)
    ap.add_argument("--max-time", type=float, default=40.0)
    ap.add_argument("--scenes", default=os.path.join(ROOT, "benchmark", "scenes.json"))
    a = ap.parse_args()

    with open(a.scenes) as fh:
        scenes = json.load(fh)["scenes"]
    match = [s for s in scenes if s["tier"] == a.tier and s["index"] == a.index]
    if not match:
        raise SystemExit(f"no tier-{a.tier} scene with index {a.index}")
    scene = match[0]

    print(f"tier {a.tier} #{a.index}: target {scene['target']} start {scene['start_pose']}")
    m, _ = rollout(make_policy(a.policy), scene, max_time=a.max_time, use_gui=True)
    print(f"\noutcome: {m['outcome']} | final {m['final_err_cm']:.1f} cm | "
          f"closest {m['min_err_cm']:.1f} cm | time {m['time_s']:.1f} s")


if __name__ == "__main__":
    run_main(main, HERE)
