"""One table comparing every policy tested on the same benchmark scenes.

    python compare.py --results results50                               # every policy, tier 1
    python compare.py --results results50 --tier 2 --policies expert vK_kick
    python compare.py --results results50 --markdown                    # table for the README / report

Columns: success within 1 cm (the task's tolerance) with a 95% Wilson confidence range (how much
of the number could be luck with this many scenes), within 2 cm and 5 cm (looser tolerances, judged
on the closest approach), crashes, timeouts, median final distance, mean time to reach.
"""
import argparse
import csv
import glob
import os

import numpy as np


def wilson(k, n, z=1.96):
    """95% confidence range for a success rate (Wilson interval): how much could be luck."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)

from quiet import run_main

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", type=int, default=1)
    ap.add_argument("--policies", nargs="*", default=None, help="names to include, in this order")
    ap.add_argument("--results", default=os.path.join(os.path.dirname(os.path.dirname(HERE)), "results", "new_runs"))
    ap.add_argument("--out", default=None, help="also save the table as CSV")
    ap.add_argument("--markdown", action="store_true", help="print a markdown table (for the README / report)")
    a = ap.parse_args()

    files = {os.path.basename(f)[: -len(f"_tier{a.tier}.csv")]: f
             for f in glob.glob(os.path.join(a.results, f"*_tier{a.tier}.csv"))}
    names = a.policies or sorted(files, key=lambda n: (n != "expert", n))
    rows = []
    for n in names:
        if n not in files:
            print(f"note: no tier-{a.tier} results for {n}")
            continue
        with open(files[n]) as fh:
            r = list(csv.DictReader(fh))
        ok = [x for x in r if x["success"] == "True"]
        out = [x["outcome"] for x in r]
        crashes = sum(o.startswith("crashed") or o in ("tipped_over", "unstable") for o in out)
        closest = [float(x["min_err_cm"]) for x in r]
        lo, hi = wilson(len(ok), len(r))
        rows.append({
            "policy": n, "scenes": len(r), "reached": len(ok),
            "ci95": f"{100 * lo:.0f}-{100 * hi:.0f}%",
            "within_2cm": sum(c <= 2.0 for c in closest), "within_5cm": sum(c <= 5.0 for c in closest),
            "crashes": crashes, "timeouts": out.count("timeout"),
            "median_final_cm": round(float(np.median([float(x["final_err_cm"]) for x in r])), 1),
            "median_closest_cm": round(float(np.median([float(x["min_err_cm"]) for x in r])), 1),
            "time_to_reach_s": round(float(np.mean([float(x["time_s"]) for x in ok])), 1) if ok else None,
        })
    if a.markdown:
        print(f"Tier {a.tier}\n")
        print("| policy | within 1 cm | 95% range | within 2 cm | within 5 cm | crashes | timeouts | median final (cm) | time to reach (s) |")
        print("|---|---|---|---|---|---|---|---|---|")
        for x in rows:
            n = x["scenes"]
            print(f"| {x['policy']} | {x['reached']}/{n} | {x['ci95']} | {x['within_2cm']}/{n} | {x['within_5cm']}/{n} | "
                  f"{x['crashes']} | {x['timeouts']} | {x['median_final_cm']:.1f} | {x['time_to_reach_s'] if x['time_to_reach_s'] else '-'} |")
    else:
        head = (f"{'policy':22s} {'1 cm':>7s} {'(95% range)':>12s} {'2 cm':>6s} {'5 cm':>6s} {'crashes':>8s} "
                f"{'timeouts':>9s} {'final cm':>9s} {'reach s':>8s}")
        print(head)
        print("-" * len(head))
        for x in rows:
            n = x["scenes"]
            print(f"{x['policy']:22s} {x['reached']:>3d}/{n:<3d} {x['ci95']:>12s} {x['within_2cm']:>3d}/{n:<2d} "
                  f"{x['within_5cm']:>3d}/{n:<2d} {x['crashes']:>8d} {x['timeouts']:>9d} "
                  f"{x['median_final_cm']:>9.1f} {str(x['time_to_reach_s'] or '-'):>8s}")
    if a.out and rows:
        with open(a.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    run_main(main, HERE)
