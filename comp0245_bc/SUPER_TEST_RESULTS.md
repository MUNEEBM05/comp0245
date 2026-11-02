# Super test results and final decisions (1–2 Nov)

## 5. Update 1 Nov: BO final run (replaces section 3's decision)

- The overnight BO run (31 Oct) won under a rule fixed in advance: replace R only if fresh ≥86% and no other test drops more than 2 points. It won, so the final arm is `bo_t03` + Kalman (kfv).
- Caveat: one training seed for the final BO model. BO tuned for driving success, not training loss.
- Report sentence: "We used Bayesian optimisation to tune training for driving success rather than training loss, then confirmed it on unseen scenes with a rule fixed in advance."

## 6. Super test (1–2 Nov): one rule, one LiDAR, all team work merged

- Everyone's navigation was merged into one codebase and given the same LiDAR (0.30 m high, rays start 0.40 m out). Success = held within 1 cm for 5 s; 40 s limit, 90 s with obstacles. 610 scenes per system. S1 and S5 ran on David's laptop, S2–S4 on the server; all 60 result files checked (50 or 30 rows each).

  | System | Fresh | Hard targets | Awkward starts | Noise 0.01 | Noise+delay | Obstacles (150) | Box | Door | Overall |
  |---|---|---|---|---|---|---|---|---|---|
  | S1 Expert + Abdullah's planner (ceiling) | 100 | 100 | 98 | 100 | 76 | 99 | 100 | 100 | 96.7 |
  | S2 BO+KF + David's nav (VFH/Bug 2) | 90 | 96 | 92 | 90 | 4 | 98 | 94 | 84 | 81.0 |
  | **S3 BO+KF + Abdullah's A* planner** | 90 | 92 | 82 | 92 | 16 | 87 | 98 | 94 | **81.3** |
  | S4 BO+KF + Driver A (learned, brake) | 72 | 86 | 62 | 66 | 8 | 71 | 90 | 94 | 68.7 |
  | S5 BO+KF + planner + camera (target found by camera) | 76 | 44 | 72 | 52 | 14 | 63 | 48 | 74 | 55.4 |

- **Final system:** S3 by the pre-set rule (best overall), but S2 and S3 are a tie (0.3 points). S2 is better in clutter and awkward starts, S3 on box, door and delay.
- **Findings:** learned driving (Driver A) is about 12 points behind hand-built navigation; the camera works but cannot see high targets (44%); noise + delay breaks every learned arm (4–16%), while the expert keeps 76%.
- **Caveats:** box and door were used while building David's nav; Abdullah's brake was added after seeing test crashes; the BO arm was never trained for delay; one seed.
- Report sentence: "Under one shared test, two navigation methods tied at 81%; we chose by a rule fixed beforehand. Fully learned driving and camera-only targeting work but cost 12–26 points, and command delay remains the main open weakness."
