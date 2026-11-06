#!/bin/bash
# One command to check and reproduce the report. Run it from this folder, inside RoboEnv's pixi shell
# (README, section 1):
#
#   bash reproduce.sh check   (~10 s)   setup check, every trained model loads, and every number in the
#                                    report is recounted from the saved per-scene result files
#   bash reproduce.sh quick   (~5 min)  check + rerun K, R, the BO network + Kalman filter and the final
#                                    system S3 on the first 10 scenes, compared scene by scene with the
#                                    saved results
#   bash reproduce.sh full    (~3 h)    check + rerun the final system S3 on the whole super test (500 trials,
#                                    the same commands as the original), scored by evaluation/super_table.py
#
#   bash reproduce.sh list               every model in the report, by its report name
#   bash reproduce.sh run                asks which model (K, P, R, BO, S1-S5, ModelA ...) and how many scenes
#   bash reproduce.sh run S3 10          runs it straight away and compares with the saved result
#   bash reproduce.sh watch S3           opens the simulator window on one drive
#
# Reruns go to results/reproduced/ and never overwrite the saved results. Set WORKERS=8 to use more cores.
set -u
cd "$(dirname "$0")"
PY=${PY:-python3}
WORKERS=${WORKERS:-4}
export WORKERS
OUT=results/reproduced
MODE=${1:-check}
case "$MODE" in check|quick|full|list|run|watch) ;;
  *) echo "usage: bash reproduce.sh [check | quick | full | list | run [NAME [N]] | watch NAME]"; exit 2;; esac

line() { printf '\n== %s\n' "$*"; }
fail() { echo "STOP: $*"; exit 1; }

[ "$MODE" = list ] && exec $PY evaluation/run_model.py list

# ---------------------------------------------------------------- check
line "1/3 setup"
$PY -c "import numpy, pybullet, pinocchio, matplotlib" 2>/dev/null \
  || fail "numpy / pybullet / pinocchio not found. Open RoboEnv's environment first: pixi shell --manifest-path /path/to/RoboEnv/pixi.toml"
[ -e models/robotnik ] || [ -n "$(ls models 2>/dev/null)" ] \
  || fail "the robot models are missing. Link them: ln -s /path/to/RoboEnv/models models"
echo "python: $($PY -c 'import sys; print(sys.executable)')"
echo "robot models: ok"

# run / watch one model by its report name (only needs the setup check above)
if [ "$MODE" = run ] || [ "$MODE" = watch ]; then
  shift
  exec $PY evaluation/run_model.py "$MODE" "$@"
fi

line "2/3 every trained model loads"
$PY - <<'EOF' || fail "a model file did not load"
import os, pickle, sys, numpy as np
bad = 0
files = sorted(f for f in os.listdir("trained") if f.endswith((".npz", ".pkl")) and f != "gp_R.pkl")
for f in files:
    try:
        if f.endswith(".npz"):
            z = np.load(os.path.join("trained", f), allow_pickle=True); [z[k] for k in z.files]
        else:
            pickle.load(open(os.path.join("trained", f), "rb"))
    except Exception as e:
        bad += 1; print("FAILED", f, e)
print(f"{len(files) - bad} of {len(files)} model files load")
sys.exit(1 if bad else 0)
EOF

line "3/3 every report number, recounted from the saved result files"
$PY analysis/report_numbers.py || fail "a report number no longer matches its result file (see above)"
[ "$MODE" = check ] && { echo; echo "check passed."; exit 0; }

# ---------------------------------------------------------------- reruns
export ROBOENV_HEADLESS=1 NAV_STUCK=0 OBJ_TIME=90   # the settings every saved result was made with
ev() {  # ev <policy> <outdir> <args...>
  local policy=$1 out=$2; shift 2
  $PY evaluation/evaluate.py --policy "$policy" --workers "$WORKERS" --plots 0 --hold 5 --out "$out" "$@" | grep -- "->"
}

if [ "$MODE" = quick ]; then
  Q=$OUT/quick
  rm -rf "$Q"
  line "quick rerun: first 10 scenes of each, held 5 s"
  echo "K, T1";                       ev trained/vK_kick.npz        "$Q" --tiers 1 --n 10
  echo "R, T1";                       ev trained/vR_ik.npz          "$Q" --tiers 1 --n 10
  echo "BO network + Kalman, T1";     ev trained/bo_t03.npz:kfv     "$Q" --tiers 1 --n 10
  echo "S3 final system, fresh";      ev planner:trained/bo_t03.npz:kfv "$Q/fresh" --scenes benchmark/scenes_fresh.json --tiers 1 --n 10
  line "compared with the saved results, scene by scene"
  $PY - <<'EOF'
import csv
pairs = [("K, T1", "results/1_main_models/K/tier1/vK_kick_hold5_tier1.csv", "results/reproduced/quick/vK_kick_hold5_tier1.csv"),
         ("R, T1", "results/1_main_models/R/tier1/vR_ik_hold5_tier1.csv", "results/reproduced/quick/vR_ik_hold5_tier1.csv"),
         ("BO + Kalman, T1", "results/1_main_models/R-BO/bo_t03_kfv_hold5_tier1.csv", "results/reproduced/quick/bo_t03_kfv_hold5_tier1.csv"),
         ("S3, fresh", "results/3_final_tests/super_test/fresh_scenes/planner_bo_t03_kfv_hold5_tier1.csv",
          "results/reproduced/quick/fresh/planner_bo_t03_kfv_hold5_tier1.csv")]
for name, saved, new in pairs:
    s = {r["index"]: r["outcome"] for r in csv.DictReader(open(saved))}
    n = {r["index"]: r["outcome"] for r in csv.DictReader(open(new))}
    same = sum(s[i] == n[i] for i in n)
    print(f"{name:18s} held: saved {sum(s[i] == 'reached' for i in n)}/{len(n)}, rerun {sum(v == 'reached' for v in n.values())}/{len(n)}"
          f"   same outcome in {same}/{len(n)} scenes")
EOF
  echo
  echo "quick rerun done (results in $Q). A scene or two can differ between computers; see README, section 10."
  exit 0
fi

# full: the final system S3 on the whole super test, with the same commands and settings as the saved results
S=$OUT/super_test
P=planner:trained/bo_t03.npz:kfv
line "full rerun of the final system S3 on the super test (500 trials, about 3 h with 4 workers)"
echo "1 fresh";       ev $P $S/fresh_scenes    --scenes benchmark/scenes_fresh.json --tiers 1
echo "1 hard targets"; ev $P $S/benchmark_tiers --tiers 2
echo "1 awkward";     ev $P $S/benchmark_tiers --tiers 3
echo "2 noise 0.01";  ev $P $S/fresh_scenes    --scenes benchmark/scenes_fresh.json --tiers 1 --set obs_noise=0.01
echo "2 noise+delay"; ev $P $S/benchmark_tiers --tiers 4
echo "3 obstacles";   ev $P $S/obstacles       --scenes scenes_obstacles/obstacles_test.json --tiers 10 11 12 13 14 --max-time 90
echo "3 box";         ev $P $S/benchmark_tiers --tiers 1 --set objects=3
echo "3 door";        ev $P $S/benchmark_tiers --tiers 1 --set objects=4
line "rerun scored the same way as the report's super test"
$PY evaluation/super_table.py --dir $S
line "the saved super test, for comparison"
$PY evaluation/super_table.py | sed -n '1p;4p'
