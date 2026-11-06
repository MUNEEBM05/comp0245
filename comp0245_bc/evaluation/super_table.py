"""Build the super-test table straight from the CSV success columns.

    python3 evaluation/super_table.py                                   # the saved super test
    python3 evaluation/super_table.py --dir results/reproduced/super_test   # a rerun made by reproduce.sh
"""
from _paths import ROOT, find_result  # noqa: F401  (project folder; lets the code folders import each other)
import argparse, csv, os, sys
ap = argparse.ArgumentParser()
ap.add_argument("--dir", default=os.path.join(ROOT, "results", "3_final_tests", "super_test"),
                help="folder with fresh_scenes/, benchmark_tiers/ and obstacles/ subfolders")
R = ap.parse_args().dir
SYS = [("S1 expert + planner (ceiling)", "planner_expert"), ("S2 BO+KF + our nav", "bo_t03_kfv_nav"),
       ("S3 BO+KF + Abdullah planner", "planner_bo_t03_kfv"), ("S4 BO+KF + Driver A (brake)", "driverbrake_driver_A_bo_t03_kfv"),
       ("S5 BO+KF + planner + camera", "selfnav_cam_planner_bo_t03_kfv")]
COLS = [("Fresh", "fresh_scenes", "{n}_hold5_tier1.csv"), ("Hard tgt", "benchmark_tiers", "{n}_hold5_tier2.csv"),
        ("Awkward", "benchmark_tiers", "{n}_hold5_tier3.csv"), ("Noise .01", "fresh_scenes", "{n}_obs_noise0.01_hold5_tier1.csv"),
        ("Noise+delay", "benchmark_tiers", "{n}_hold5_tier4.csv"), ("Obst x150", "obstacles", ["{n}_hold5_tier%d.csv" % t for t in range(10, 15)]),
        ("Box", "benchmark_tiers", "{n}_objects3_hold5_tier1.csv"), ("Door", "benchmark_tiers", "{n}_objects4_hold5_tier1.csv")]
STAGES = {"Stage1 acc": [0, 1, 2], "Stage2 robust": [3, 4], "Stage3 obst": [5, 6, 7]}

def score(d, pats, n):
    pats = pats if isinstance(pats, list) else [pats]
    ok = tot = 0
    for p in pats:
        f = os.path.join(R, d, p.format(n=n))
        if not os.path.exists(f): return None
        rows = list(csv.DictReader(open(f)))
        k = [c for c in rows[0] if "succ" in c][0]
        ok += sum(r[k] in ("1", "True", "true", "1.0") for r in rows); tot += len(rows)
    return 100 * ok / tot, ok, tot

out = []
for lab, n in SYS:
    v = [score(d, p, n) for _, d, p in COLS]
    if not any(v):
        continue                       # a system that was not run (e.g. a partial rerun)
    row = [lab] + [("%.0f" % x[0]) if x else "-" for x in v]
    for st, idx in STAGES.items():
        xs = [v[i][0] for i in idx if v[i]]
        row.append("%.0f" % (sum(xs) / len(xs)) if len(xs) == len(idx) else "-")
    full = all(v); row.append("%.1f" % (sum(x[0] for x in v) / len(v)) if full else "-")
    out.append(row)
hdr = ["System"] + [c[0] for c in COLS] + list(STAGES) + ["Overall"]
print(" | ".join(hdr))
for r in out: print(" | ".join(r))
if not out:
    sys.exit(f"no super-test results found in {R}")
