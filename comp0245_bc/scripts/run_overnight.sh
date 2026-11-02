#!/bin/bash
# Overnight chain (31 Oct). Every step skips work already done, so this script can simply be restarted.
#  1. wait for the stuck-detector test (run_nav.sh), then KEEP the stuck detector only if it beats the original
#     layer on the door without losing on normal scenes; otherwise switch it off (NAV_STUCK=0) and put the
#     original layer's results back as the headline. The decision is written to nav_decision.txt.
#  2. Kalman filter tests (run_kf.sh)
#  3. LiDAR student: teacher score, recording, training model S, testing S (run_student.sh)
cd /home/claude/comp0245_bc
while pgrep -f "run_nav[.]sh" > /dev/null; do sleep 30; done
if [ ! -f nav_decision.txt ]; then
  [ -f results50/vR_ik_filt20noqd_nav_hold5_tier1.csv ] || ./run_nav.sh >> nav_v3.log 2>&1
  python3 - <<'PY' > nav_decision.txt
import json
r = lambda p: json.load(open(p))["1"]["success_rate"]
v1, v3 = "results50/nav_v1_original/vR_ik_filt20noqd_nav_", "results50/vR_ik_filt20noqd_nav_"
d1, d3 = r(v1 + "objects4_hold5_summary.json"), r(v3 + "objects4_hold5_summary.json")
b1, b3 = r(v1 + "objects3_hold5_summary.json"), r(v3 + "objects3_hold5_summary.json")
n1, n3 = r(v1 + "hold5_summary.json"), r(v3 + "hold5_summary.json")
keep = d3 > d1 and n3 >= n1 - 0.02 and b3 >= b1 - 0.02
print("KEEP" if keep else "REVERT", f"door {d1:.0%}->{d3:.0%} box {b1:.0%}->{b3:.0%} normal {n1:.0%}->{n3:.0%}")
PY
  if grep -q REVERT nav_decision.txt; then
    mkdir -p results50/nav_v3_stuck_rejected
    mv results50/vR_ik_filt20noqd_nav_* results50/nav_v3_stuck_rejected/
    cp results50/nav_v1_original/* results50/
  fi
fi
grep -q REVERT nav_decision.txt && export NAV_STUCK=0 || export NAV_STUCK=1
echo "== nav decision: $(cat nav_decision.txt)  (NAV_STUCK=$NAV_STUCK)  $(date +%H:%M)"
./run_kf.sh > kf.log 2>&1; echo "== kalman done $(date +%H:%M)"; cat kf.log | grep -- "->\|best"
./run_student.sh >> student.log 2>&1; echo "== student done $(date +%H:%M)"
grep -q "== done" student.log && echo ALL_DONE > overnight_done.txt
