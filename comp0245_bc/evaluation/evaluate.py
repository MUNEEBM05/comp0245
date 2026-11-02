"""Run a policy on the frozen benchmark scenes and summarise how it does (closed-loop evaluation).

    python evaluation/evaluate.py --policy expert --tiers 1 --n 10                 # the expert, first 10 tier-1 scenes
    python evaluation/evaluate.py --policy trained/mlp_v0.npz --tiers 1 --n 10     # our model, same scenes
    python evaluation/evaluate.py --policy trained/mlp_v0.npz --tiers 1 2 3 4 --workers 3   # full benchmark

Writes <out>/<policy>_tier<k>.csv (one row per scene: outcome, final and closest distance, time,
jerk, decision time, where the base ended) and <out>/<policy>_summary.json, plus plots of the
first few drives. The policy argument is 'expert', a path to a trained model, or
'<model>:mlp_arm' / '<model>:mlp_wheels' for the diagnostic hybrids (see rollout.py).
"""
from _paths import ROOT, find_result  # noqa: F401  (project folder; lets the code folders import each other)
import argparse
import csv
import json
import os
import time
from multiprocessing import Pool

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from quiet import run_main

HERE = ROOT


def _init_worker():
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _one(job):
    spec, scene, max_time, plot_path, hold = job
    from rollout import make_policy, rollout
    scene = dict(scene)
    try:
        metrics, traj = rollout(make_policy(spec), scene, max_time=max_time, hold_s=hold)
    except Exception as exc:
        return {"tier": scene["tier"], "index": scene["index"], "seed": scene["seed"],
                "outcome": f"error: {type(exc).__name__}: {exc}", "success": False}
    if plot_path:
        try:
            plot_drive(metrics, traj, scene, plot_path)
        except OSError:
            pass
    return metrics


def plot_drive(m, traj, scene, path):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    bp, ee = traj["base_pos"], traj["ee_pos"]
    ax[0].plot(bp[:, 0], bp[:, 1], "b-", label="base path")
    ax[0].plot(ee[:, 0], ee[:, 1], "m:", label="hand path")
    ax[0].plot(*scene["start_pose"][:2], "go", label="start")
    ax[0].plot(*scene["target"][:2], "r*", ms=14, label="target")
    ax[0].set_aspect("equal"); ax[0].grid(alpha=0.3); ax[0].legend(fontsize=8)
    ax[0].set_title(f"{m['policy']} | tier {m['tier']} #{m['index']} | {m['outcome']}")
    ax[1].semilogy(traj["t"], np.maximum(traj["err"], 1e-4) * 100, "g-")
    ax[1].axhline(1.0, color="r", ls="--", lw=0.8)
    ax[1].set_xlabel("time (s)"); ax[1].set_ylabel("hand-to-target (cm)"); ax[1].grid(alpha=0.3)
    ax[2].plot(traj["t"], traj["torque_arm"], lw=0.8)
    ax[2].set_xlabel("time (s)"); ax[2].set_title("arm torques (Nm)"); ax[2].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=85)
    plt.close(fig)


def summarise(rows):
    ok = [r for r in rows if r.get("success")]
    n = len(rows)
    def mean(key, subset):
        vals = [r[key] for r in subset if key in r]
        return float(np.mean(vals)) if vals else float("nan")
    outcomes = {}
    for r in rows:
        outcomes[r["outcome"]] = outcomes.get(r["outcome"], 0) + 1
    return {
        "scenes": n, "success_rate": len(ok) / n if n else 0.0,
        "median_final_err_cm": float(np.median([r["final_err_cm"] for r in rows if "final_err_cm" in r]) if rows else np.nan),
        "mean_time_to_reach_s": mean("time_s", ok),
        "torque_jerk": mean("torque_jerk", rows), "wheel_jerk": mean("wheel_jerk", rows),
        "think_us_per_step": mean("think_us_per_step", rows), "outcomes": outcomes,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", required=True, help="'expert' or path to a trained .npz")
    ap.add_argument("--tiers", type=int, nargs="+", default=[1])
    ap.add_argument("--n", type=int, default=None, help="only the first n scenes of each tier")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--max-time", type=float, default=40.0)
    ap.add_argument("--plots", type=int, default=3, help="plot the first k drives of each tier")
    ap.add_argument("--scenes", default=os.path.join(HERE, "benchmark", "scenes.json"))
    ap.add_argument("--out", default=os.path.join(HERE, "results", "new_runs"))
    ap.add_argument("--hold", type=float, default=0.0,
                    help="stricter success: the hand must stay within 1 cm this many seconds (results get _hold<s>)")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="Stage B: add one disturbance to every scene, e.g. obs_noise=0.01, delay_steps=2, "
                         "torque_noise=2, payload_kg=0.5 (results get a matching suffix)")
    ap.add_argument("--only", choices=["noise", "delay"],
                    help="diagnosis: keep only this part of each scene's disturbance (tier 4: noise vs delay)")
    a = ap.parse_args()

    wrapped = ":filt" in a.policy or a.policy.startswith("ens:") or a.policy.endswith(":nav") or a.policy.endswith(":mapnav") or a.policy.endswith(":kf") or a.policy.endswith(":kfv") or a.policy == "planner" or a.policy.startswith(("planner:", "driver:", "driverbrake:", "cam:"))
    policy_file = a.policy.rsplit(":", 1)[0] if a.policy.endswith((":mlp_arm", ":mlp_wheels")) else a.policy
    if not wrapped and a.policy not in ("expert", "markov_expert") and not os.path.exists(policy_file):
        print(f"Policy file not found: {a.policy}")
        return
    if not os.path.exists(a.scenes):
        print(f"No benchmark found at {a.scenes}. Run make_benchmark.py first.")
        return
    with open(a.scenes) as fh:
        scenes = json.load(fh)["scenes"]
    if wrapped:
        from rollout import make_policy as _mp
        name = _mp(a.policy).name
    else:
        name = a.policy if a.policy in ("expert", "markov_expert") else os.path.splitext(os.path.basename(policy_file))[0]
    if policy_file.endswith(".pkl"):
        print("note: a random forest predicts far slower than an MLP; each drive takes longer")
    if policy_file != a.policy:
        name += "_" + a.policy.rsplit(":", 1)[1]
    for kv in a.set:
        k, v = kv.split("=")
        for s in scenes:
            s["disturbance"] = dict(s.get("disturbance") or {}, **{k: (int(v) if k == "delay_steps" else float(v))})
        name += f"_{k}{v}"
    if a.hold > 0:
        name += f"_hold{a.hold:g}"
    if a.only:
        keep = {"noise": "obs_noise", "delay": "delay_steps"}[a.only]
        for s in scenes:
            s["disturbance"] = {k: v for k, v in (s.get("disturbance") or {}).items() if k == keep}
        name += f"_{a.only}only"
    os.makedirs(a.out, exist_ok=True)

    summaries = {}
    for tier in a.tiers:
        chosen = [s for s in scenes if s["tier"] == tier][: a.n]
        jobs = [(a.policy, s, a.max_time,
                 os.path.join(a.out, f"{name}_t{tier}_{s['index']:02d}.png") if i < a.plots else None, a.hold)
                for i, s in enumerate(chosen)]
        print(f"\n{name}: tier {tier}, {len(jobs)} scenes")
        t0, rows = time.time(), []

        def report(m):
            rows.append(m)
            extra = f"err {m['final_err_cm']:6.1f} cm | t {m['time_s']:5.1f} s" if "final_err_cm" in m else ""
            print(f"  [{len(rows):3d}/{len(jobs)}] scene {m['index']:2d}: {m['outcome']:14s} {extra}", flush=True)

        if a.workers > 1:
            pool = Pool(a.workers, initializer=_init_worker)
            try:
                for m in pool.imap_unordered(_one, jobs):
                    report(m)
                pool.close()
            except KeyboardInterrupt:
                pool.terminate()
                raise
            finally:
                pool.join()
        else:
            for j in jobs:
                report(_one(j))

        rows.sort(key=lambda r: r["index"])
        csv_path = os.path.join(a.out, f"{name}_tier{tier}.csv")
        keys = sorted({k for r in rows for k in r})
        try:
            with open(csv_path, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=keys)
                w.writeheader()
                w.writerows(rows)
        except PermissionError:
            print(f"  note: could not write {csv_path}: is it open in Excel?")
        s = summarise(rows)
        summaries[tier] = s
        print(f"  -> success {100 * s['success_rate']:.0f}% | median final error {s['median_final_err_cm']:.1f} cm | "
              f"time to reach {s['mean_time_to_reach_s']:.1f} s | {s['outcomes']} | "
              f"{(time.time() - t0) / 60:.1f} min")

    with open(os.path.join(a.out, f"{name}_summary.json"), "w") as fh:
        json.dump(summaries, fh, indent=1)


if __name__ == "__main__":
    run_main(main, HERE)
