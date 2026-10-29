"""DAgger: let OUR model drive, and record what the expert WOULD have done in every state it reaches.

Why: a model trained only on the expert's own drives never sees the states its own small mistakes
lead to, so the mistakes compound (Ross, Gordon & Bagnell, AISTATS 2011). DAgger fixes this by
collecting data where the model actually goes:

    iteration i:  1. the current model drives a set of new training scenes
                  2. at every step the expert is asked "what would you do here?" (its answer is
                     recorded as the label but NOT executed, unless --beta mixes it in)
                  3. all data so far (clean + nudged + every DAgger round) is used to retrain

    python dagger.py --policy trained/vK_kick.npz --start 41000 --end 41100 --out data/runs_hg1 --workers 2
    python train.py --name vL_dagger1 ... --extra-train data/runs_nudged data/runs_reach data/runs_kick data/runs_dagger1

The expert has memory (its navigation phase and its creeping arm reference). It is run "in the
shadow" alongside the model from the very first step, so its memory evolves on the states the model
visits: exactly the expert we would get if it were handed the controls at that moment. Its
controllers are the template's, unchanged.

Recordings use the same file format as expert_episode.py (DATA_FORMAT.md), so train.py and the
feature code read them unchanged. Labels (torque_arm, wheel_cmds, q_des_arm, ...) are the EXPERT's
commands; the executed commands (the model's, or the expert's when --beta picks it) are stored as
torque_arm_sent / wheel_cmds_sent. meta["dagger"] = True tells train.py to keep every DAgger run,
including the ones where the model failed: those are exactly the states we want labelled.
Safe to stop and restart: runs whose file already exists are skipped.

Two modes (--mode):
    dagger  classic DAgger: the model always drives (unless --beta), every row is labelled and kept.
            RESULT: made model K much WORSE (36/50 -> 1/50). The expert's memory is not visible in the
            measurements: when the model starts reaching while the expert still thinks it is turning
            ("orient"), the expert's label is "pull the arm home", for up to 20 s, until its 25 s
            navigation timeout flips it to "reach" in the very same states. 15% of the rows carried such
            labels that no model can predict from what it sees.
    hg      HG-DAgger (Kelly et al., ICRA 2019), the interventional variant: the model drives, the expert
            takes over when the two disagree (or the model reaches before the expert would), steers back
            on track, then hands back. Only the rows where the EXPERT was driving are used for training
            (saved as the train_mask array), so every label comes from an expert whose memory matches
            the situation it is actually in.
"""
import argparse
import json
import os
import signal
import time
import traceback
from multiprocessing import Pool

import numpy as np

from quiet import run_main

HERE = os.path.dirname(os.path.abspath(__file__))
SEGMENT_S = 0.5      # with --beta, who drives is re-decided every 0.5 s (a coin flip per segment)


def _ignore_ctrl_c():
    signal.signal(signal.SIGINT, signal.SIG_IGN)


# HG-DAgger takeover rule (mode "hg"): the expert takes the controls when, for 0.1 s in a row, the model's
# command is far from the expert's, or the model's arm has left its tucked pose while the expert has not
# finished parking. It keeps them for at least 1 s and until the model has agreed with it for 0.2 s.
TAKEOVER_TORQUE_NM = 3.0         # mean |torque difference| over the 7 joints
TAKEOVER_WHEEL = 1.0             # mean |wheel-speed difference|, rad/s
ARM_OUT_RAD = 0.1                # "arm has left its tucked pose"
TRIGGER_S, MIN_HOLD_S, AGREE_S = 0.1, 1.0, 0.2


def dagger_episode(policy_path, seed, out_path=None, beta=0.0, max_time=30.0, log_hz=100, mode="dagger",
                   labeller="markov"):
    """One DAgger drive on training scene `seed`. Returns the meta dict (and writes out_path)."""
    from expert_episode import (ARM, ARM_REACH_LIMIT, BASE_NAV_TIMEOUT, PHASE_CODES, REACH_TOLERANCE,
                                STANDOFF_POINTS, STANDOFF_Z_POINTS, pin)
    from robotnik_arm_control import (ArmControllerParams, BaseControllerParams, BaseNavState, arm_controller,
                                      base_controller, quaternion2bearing)
    from rollout import World, make_policy
    from samplers import training_scene

    target, start = training_scene(seed)
    rng = np.random.default_rng(seed)
    wall0 = time.time()
    world = World(start, target)
    sim, c = world.sim, world.client
    try:
        learner = make_policy(policy_path)
        learner.reset(world, target)

        # The expert, exactly as in expert_episode.py, but only consulted (not obeyed)
        standoff = float(np.interp(target[2], STANDOFF_Z_POINTS, STANDOFF_POINTS))
        tb = np.arctan2(target[1], target[0]) if np.linalg.norm(target[:2]) > 1e-4 else 0.0
        goal = target[:2] - standoff * np.array([np.cos(tb), np.sin(tb)])
        nav = BaseNavState()
        base_params = BaseControllerParams(nav_timeout=BASE_NAV_TIMEOUT)
        arm_params = ArmControllerParams(reach_limit=ARM_REACH_LIMIT)
        q_hold = world.init_joint_angles[ARM].copy()
        q_des = q_hold.copy()
        if labeller == "markov":
            from markov_expert import MarkovExpert
            mx = MarkovExpert(world, target, goal, tb, base_params, arm_params, q_hold, world.dt)

        dt = world.dt
        log_every = max(1, int(round(1.0 / (log_hz * dt))))
        seg_steps = max(1, int(round(SEGMENT_S / dt)))
        max_steps = int(round(max_time / dt))
        log = {k: [] for k in [
            "t", "base_pos", "base_quat", "base_bearing", "q_mes", "qd_mes", "ee_pos", "target",
            "phase", "nav_done", "q_des_arm", "qd_des_arm", "tau_gravity",
            "torque_arm", "wheel_cmds", "torque_arm_sent", "wheel_cmds_sent",
            "contact_ground", "contact_self", "expert_drives", "train_mask",
        ]}
        disagree_steps = agree_steps = hold_steps = 0
        n_takeovers = 0
        t, step, outcome, err, min_err = 0.0, 0, None, np.inf, np.inf
        expert_drives, expert_steps = False, 0

        while True:
            # --- Measure (true state; DAgger is about the model's own mistakes, no sensor noise) ---
            base_pos = np.array(sim.GetBasePosition())
            base_quat = np.array(sim.GetBaseOrientation())
            bearing = quaternion2bearing(base_quat[3], base_quat[0], base_quat[1], base_quat[2])
            q_mes = np.array(sim.GetMotorAngles(0))
            qd_mes = np.array(sim.GetMotorVelocities(0))
            ee, q_pin = world.forward_kinematics(base_pos, base_quat, q_mes)
            err = float(np.linalg.norm(target - ee))
            min_err = min(min_err, err)

            # --- Ask the expert (the LABEL) ---------------------------------------------------
            if labeller == "markov":   # memory-free expert: same controllers, phase read from the state
                wheels_e, torque_e, q_des, qd_des, phase_before = mx.label(
                    base_pos, bearing, q_mes[ARM], qd_mes[ARM], q_pin, ee)
                nav.done = phase_before == "done"
            else:                      # the template expert, run in the shadow with its own memory
                phase_before = "done" if nav.done else nav.phase
                wheels_e = base_controller(nav, base_pos, bearing, goal, tb, t, base_params)
                torque_e, q_des, qd_des = arm_controller(
                    world.dyn_model, q_pin, q_mes[ARM], qd_mes[ARM], q_des, q_hold, target, ee,
                    world.ee_frame_id, world.shoulder_frame_id, world.arm_lower, world.arm_upper, dt,
                    arm_params, reach_active=nav.done)

            # --- Ask the model (what usually gets EXECUTED) ------------------------------------
            obs = {"t": t, "base_pos": base_pos, "base_quat": base_quat, "base_bearing": bearing,
                   "q_mes": q_mes, "qd_mes": qd_mes, "ee_pos": ee, "q_pin": q_pin}
            wheels_l, torque_l = learner.act(obs)
            if mode == "hg":
                gap_t = float(np.abs(torque_e - torque_l).mean())
                gap_w = float(np.abs(wheels_e - wheels_l).mean())
                too_early = (not nav.done) and float(np.abs(q_mes[ARM] - q_hold).max()) > ARM_OUT_RAD
                disagree = gap_t > TAKEOVER_TORQUE_NM or gap_w > TAKEOVER_WHEEL or too_early
                disagree_steps = disagree_steps + 1 if disagree else 0
                agree_steps = 0 if disagree else agree_steps + 1
                if not expert_drives and disagree_steps * dt >= TRIGGER_S:
                    expert_drives, hold_steps, n_takeovers = True, 0, n_takeovers + 1
                elif expert_drives:
                    hold_steps += 1
                    if hold_steps * dt >= MIN_HOLD_S and agree_steps * dt >= AGREE_S:
                        expert_drives = False
            elif beta > 0 and step % seg_steps == 0:
                expert_drives = bool(rng.random() < beta)
            sent = np.concatenate((wheels_e, torque_e) if expert_drives else (wheels_l, torque_l))
            expert_steps += int(expert_drives)
            if not np.all(np.isfinite(sent)):
                outcome = "unstable"
                break

            if step % log_every == 0:
                tau_g = pin.computeGeneralizedGravity(world.dyn_model.pin_model, world.dyn_model.pin_data,
                                                      q_pin)[6:13]
                hit = world.arm_collision()
                for k, v in (("t", t), ("base_pos", base_pos), ("base_quat", base_quat),
                             ("base_bearing", bearing), ("q_mes", q_mes), ("qd_mes", qd_mes), ("ee_pos", ee),
                             ("target", target), ("phase", PHASE_CODES[phase_before]), ("nav_done", nav.done),
                             ("q_des_arm", q_des.copy()), ("qd_des_arm", qd_des.copy()), ("tau_gravity", tau_g),
                             ("torque_arm", torque_e), ("wheel_cmds", wheels_e),
                             ("torque_arm_sent", sent[4:]), ("wheel_cmds_sent", sent[:4]),
                             ("contact_ground", hit == "ground"), ("contact_self", hit == "base"),
                             ("expert_drives", expert_drives),
                             ("train_mask", expert_drives if mode == "hg" else True)):
                    log[k].append(v)
                if hit:
                    outcome = f"crashed_{hit}"
                    break
                roll, pitch, _ = c.getEulerFromQuaternion(base_quat)
                if abs(roll) > 0.5 or abs(pitch) > 0.5:
                    outcome = "tipped_over"
                    break
            if err < REACH_TOLERANCE:
                outcome = "reached"
                break
            if step >= max_steps:
                outcome = "timeout"
                break

            world.cmd.SetControlCmd(np.concatenate((sent, np.zeros(6))), world.control_modes)
            sim.Step(world.cmd, "torque")
            t += dt
            step += 1
    finally:
        world.close()

    arrays = {}
    for k, v in log.items():
        a = np.asarray(v)
        arrays[k] = a if a.dtype == bool else (a.astype(np.int8) if k == "phase" else a.astype(np.float32))
    meta = {
        "seed": int(seed), "target": [float(v) for v in target], "start_pose": [float(v) for v in start],
        "base_goal_pos": [float(v) for v in goal], "base_goal_bearing": float(tb), "standoff": standoff,
        "dagger": True, "dagger_mode": mode, "labeller": labeller, "dagger_policy": os.path.basename(policy_path), "beta": float(beta),
        "n_takeovers": int(n_takeovers),
        "expert_share": round(expert_steps / max(step, 1), 3),
        "outcome": outcome,                       # how the MODEL's drive ended (labels are the expert's)
        "final_ee_error": err, "min_ee_error": float(min_err),
        "duration": float(t), "n_sim_steps": int(step), "n_logged": int(len(log["t"])),
        "dt": float(dt), "log_every": int(log_every), "max_time": float(max_time),
        "wall_time_s": float(time.time() - wall0),
    }
    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        tmp = out_path + ".tmp.npz"
        np.savez_compressed(tmp, meta=np.array(json.dumps(meta)), **arrays)
        os.replace(tmp, out_path)
    return meta


def _one(args):
    policy, seed, out_dir, beta, max_time, mode, labeller = args
    path = os.path.join(out_dir, f"run_{seed:05d}.npz")
    if os.path.exists(path):
        return seed, "skipped", None, None
    try:
        return seed, None, dagger_episode(policy, seed, path, beta, max_time, mode=mode, labeller=labeller), None
    except Exception as exc:
        with open(os.path.join(out_dir, "failures.log"), "a", encoding="utf-8") as f:
            f.write(f"--- seed {seed} ---\n{traceback.format_exc()}\n")
        return seed, "error", None, f"{type(exc).__name__}: {exc}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", required=True, help="the model that drives (trained/*.npz)")
    ap.add_argument("--start", type=int, required=True, help="first training seed (inclusive)")
    ap.add_argument("--end", type=int, required=True, help="last training seed (exclusive)")
    ap.add_argument("--out", required=True, help="output folder, e.g. data/runs_dagger1")
    ap.add_argument("--beta", type=float, default=0.0,
                    help="chance the EXPERT drives each 0.5 s segment (0 = model always drives, classic DAgger)")
    ap.add_argument("--mode", choices=["dagger", "hg"], default="dagger",
                    help="hg = HG-DAgger, expert takes over when needed (default); dagger = classic")
    ap.add_argument("--labeller", choices=["markov", "template"], default="markov",
                    help="markov = memory-free expert (markov_expert.py, default); template = the expert with its memory")
    ap.add_argument("--max-time", type=float, default=30.0, help="simulated seconds per drive")
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()
    if a.end <= a.start:
        print(f"Nothing to do: --end ({a.end}) must be larger than --start ({a.start}).")
        return
    if not os.path.exists(a.policy):
        print(f"Model file not found: {a.policy}")
        return
    os.makedirs(a.out, exist_ok=True)
    jobs = [(a.policy, s, a.out, a.beta, a.max_time, a.mode, a.labeller) for s in range(a.start, a.end)]
    counts, t0 = {}, time.time()
    print(f"DAgger: {os.path.basename(a.policy)} drives seeds {a.start}-{a.end - 1}, expert labels -> {a.out}")

    def report(r):
        seed, status, m, e = r
        key = status or m["outcome"]
        counts[key] = counts.get(key, 0) + 1
        done = sum(counts.values())
        if m:
            extra = f"takeovers {m['n_takeovers']:2d} | closest {m['min_ee_error'] * 100:5.1f} cm | sim {m['duration']:5.1f} s | wall {m['wall_time_s']:5.1f} s"
        else:
            extra = f"{e} (details in failures.log)" if e else "already recorded"
        print(f"[{done:4d}/{len(jobs)}] seed {seed:5d}: {key:14s} {extra}", flush=True)

    if a.workers > 1:
        pool = Pool(a.workers, initializer=_ignore_ctrl_c)
        try:
            for r in pool.imap_unordered(_one, jobs):
                report(r)
            pool.close()
        except KeyboardInterrupt:
            pool.terminate()
            raise
        finally:
            pool.join()
    else:
        for j in jobs:
            report(_one(j))
    print(f"\nDone in {(time.time() - t0) / 60:.1f} min: {json.dumps(counts)}")


if __name__ == "__main__":
    run_main(main, HERE)
