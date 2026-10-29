# Benchmark results

All numbers from `results50/`, regenerate with `python compare.py --results results50 --tier N --markdown`.

Success = hand within 1 cm of the target (the template's own tolerance); 2 cm / 5 cm judged on the closest approach.

## Tier 1: every version (the ladder), ablations and model comparison


| policy | within 1 cm | 95% range | within 2 cm | within 5 cm | crashes | timeouts | median final (cm) | time to reach (s) |
|---|---|---|---|---|---|---|---|---|
| expert | 49/50 | 90-100% | 49/50 | 49/50 | 0 | 1 | 1.0 | 9.6 |
| mlp_v0 | 0/50 | 0-7% | 0/50 | 0/50 | 48 | 2 | 125.5 | - |
| mlp_v1 | 0/50 | 0-7% | 0/50 | 0/50 | 50 | 0 | 100.5 | - |
| mlp_v1_hist | 0/50 | 0-7% | 0/50 | 0/50 | 50 | 0 | 161.8 | - |
| mlp_v2 | 3/50 | 2-16% | 4/50 | 15/50 | 1 | 46 | 20.7 | 10.0 |
| mlp_v2_hist | 1/50 | 0-10% | 4/50 | 5/50 | 0 | 49 | 58.1 | 24.5 |
| vA_goal | 11/50 | 13-35% | 19/50 | 34/50 | 0 | 39 | 8.9 | 12.3 |
| vB_separate | 0/50 | 0-7% | 5/50 | 15/50 | 3 | 47 | 27.2 | - |
| vC_reachw | 1/50 | 0-10% | 1/50 | 2/50 | 5 | 44 | 31.8 | 11.0 |
| vD_all | 20/50 | 28-54% | 33/50 | 47/50 | 8 | 22 | 3.6 | 10.8 |
| vD2_reachdata | 22/50 | 31-58% | 35/50 | 45/50 | 2 | 26 | 2.5 | 10.9 |
| vD2_seed1 | 28/50 | 42-69% | 37/50 | 47/50 | 1 | 21 | 1.0 | 15.4 |
| vD2_seed2 | 19/50 | 26-52% | 32/50 | 43/50 | 4 | 27 | 3.0 | 15.2 |
| vE_integral | 32/50 | 50-76% | 45/50 | 50/50 | 3 | 15 | 1.0 | 13.1 |
| vE_seed1 | 25/50 | 37-63% | 39/50 | 47/50 | 3 | 22 | 1.1 | 11.1 |
| vK_kick | 36/50 | 58-83% | 47/50 | 50/50 | 1 | 13 | 1.0 | 10.6 |
| vK_kick_s1 | 34/50 | 54-79% | 48/50 | 50/50 | 1 | 15 | 1.0 | 11.0 |
| vG_both | 26/50 | 39-65% | 42/50 | 50/50 | 2 | 22 | 1.0 | 12.5 |
| vEw_wheelvel | 0/50 | 0-7% | 4/50 | 7/50 | 0 | 50 | 72.1 | - |
| vEw_wheelvel_s1 | 9/50 | 10-31% | 15/50 | 23/50 | 1 | 40 | 53.5 | 25.0 |
| cmp_linear | 0/50 | 0-7% | 1/50 | 2/50 | 50 | 0 | 103.9 | - |
| cmp_forest | 0/20 | 0-16% | 0/20 | 0/20 | 18 | 2 | 196.4 | - |


## Tier 2

| policy | within 1 cm | 95% range | within 2 cm | within 5 cm | crashes | timeouts | median final (cm) | time to reach (s) |
|---|---|---|---|---|---|---|---|---|
| expert | 50/50 | 93-100% | 50/50 | 50/50 | 0 | 0 | 1.0 | 11.2 |
| vK_kick | 31/50 | 48-74% | 46/50 | 50/50 | 2 | 17 | 1.0 | 12.5 |


## Tier 3

| policy | within 1 cm | 95% range | within 2 cm | within 5 cm | crashes | timeouts | median final (cm) | time to reach (s) |
|---|---|---|---|---|---|---|---|---|
| expert | 50/50 | 93-100% | 50/50 | 50/50 | 0 | 0 | 1.0 | 10.4 |
| vK_kick | 39/50 | 65-87% | 46/50 | 50/50 | 1 | 10 | 1.0 | 13.2 |


## Tier 4

| policy | within 1 cm | 95% range | within 2 cm | within 5 cm | crashes | timeouts | median final (cm) | time to reach (s) |
|---|---|---|---|---|---|---|---|---|
| expert | 41/50 | 69-90% | 43/50 | 49/50 | 9 | 0 | 1.0 | 9.4 |
| vK_kick | 0/50 | 0-7% | 1/50 | 1/50 | 22 | 28 | 60.6 | - |
