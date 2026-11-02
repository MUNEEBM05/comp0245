"""Why did each drive fail? Sorts every failed scene into one plain reason.

    python evaluation/analyse_failures.py --results results --policies vE_integral vK_kick

The reasons, in the order they are checked:
    crash         the arm hit the ground or the base (or the base tipped / the model output NaN)
    not_arrived   the base never got within 20 cm of its parking spot
    not_stopped   the base reached the spot but was still turning or driving at the end
    not_settled   base parked, hand came within 5 cm, but never within the 1 cm tolerance
    arm_off       base parked, but the hand never came close
Uses the end-of-drive measurements that rollout.py records (goal_dist_end_m, yaw_rate_end, ...).
"""
from _paths import ROOT, find_result  # noqa: F401  (project folder; lets the code folders import each other)
import argparse
import csv
import os

from quiet import run_main

HERE = ROOT

REASONS = [
    ("crash", "arm hit the ground or the base"),
    ("not_arrived", "base never reached the parking spot"),
    ("not_stopped", "base reached the spot but kept spinning or driving"),
    ("not_settled", "base parked, hand got close (<5 cm) but never settled within 1 cm"),
    ("arm_off", "base parked, but the hand never got close"),
]


def reason(r):
    o = r["outcome"]
    if o.startswith("crashed") or o in ("tipped_over", "unstable"):
        return "crash"
    g, spin, wheels = float(r["goal_dist_end_m"]), float(r["yaw_rate_end"]), float(r["wheels_end"])
    if g > 0.20 and spin < 0.15 and wheels < 0.5:
        return "not_arrived"
    if spin >= 0.15 or wheels >= 0.5:
        return "not_arrived" if g > 0.5 else "not_stopped"
    return "not_settled" if float(r["min_err_cm"]) < 5.0 else "arm_off"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policies", nargs="+", required=True)
    ap.add_argument("--tier", type=int, default=1)
    ap.add_argument("--results", default=os.path.join(HERE, "results"))
    a = ap.parse_args()

    for name in a.policies:
        path = find_result(a.results, f"{name}_tier{a.tier}.csv")
        if not os.path.exists(path):
            print(f"\n{name}: no results yet")
            continue
        with open(path) as fh:
            rows = list(csv.DictReader(fh))
        if rows and "goal_dist_end_m" not in rows[0]:
            print(f"\n{name}: tested before the failure measurements existed; re-run evaluate.py to analyse it")
            continue
        ok = [r for r in rows if r["success"] == "True"]
        print(f"\n{name}: reached {len(ok)}/{len(rows)}")
        counts = {}
        for r in rows:
            if r["success"] == "True":
                continue
            k = reason(r)
            counts.setdefault(k, []).append(r)
        for k, text in REASONS:
            if k in counts:
                ids = ", ".join(r["index"] for r in counts[k])
                print(f"  {len(counts[k]):2d} x {text}  (scenes {ids})")
        for r in rows:
            if r["success"] != "True":
                print(f"     scene {r['index']:>2s}: end {float(r['goal_dist_end_m']) * 100:5.0f} cm from parking spot, "
                      f"facing {float(r['face_err_end_deg']):5.0f} deg off, spinning {float(r['yaw_rate_end']):.2f} rad/s, "
                      f"hand closest {float(r['min_err_cm']):5.1f} cm")


if __name__ == "__main__":
    run_main(main, HERE)
