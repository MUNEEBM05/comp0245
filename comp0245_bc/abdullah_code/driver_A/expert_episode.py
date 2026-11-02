"""Run the template's expert controller for ONE episode and record it.

This is the template's robotnik_arm_task.py main loop, turned into a function so it can be
called many times with a different target, starting pose and seed. The controllers themselves
(robotnik_arm_control.py) are used unchanged: we are cloning the given expert, not a modified one.
Verified: on the template's own scene this reproduces the template exactly (reached at 8.00 s).

Three kinds of recording, all with the expert's CLEAN command as the label:
    clean           the expert drives undisturbed
    nudged          a held random offset is added to the executed command (--noise), so the robot
                    drifts off the perfect path and the expert's corrections get recorded
    kicked          brief sharp torque pulses while reaching (--kick), so recovery from unexpected
                    arm MOTION gets recorded (fixed the late arm crashes)

Usage from Python:
    from expert_episode import run_episode
    result = run_episode(target=[0.0, 1.1, 0.5], start_pose=[0.0, 0.0, 0.0], seed=0,
                         out_path="data/runs/run_00000.npz")
"""
import json
import os
import platform
import subprocess
import time

import numpy as np

from quiet import quiet_native

with quiet_native():  # pybullet prints its build time on import
    import pinocchio as pin
    import pybullet as p
    from simulation_and_control import pb, MotorCommands, PinWrapper

from robotnik_arm_control import (
    ArmControllerParams, BaseControllerParams, BaseNavState, StopCriteria,
    arm_controller, base_controller, check_stop, configure_gripper_fingers, quaternion2bearing,
)

# ------------------------------------------------------------------------------------------
# Constants copied from the template so the expert behaves exactly as given
# ------------------------------------------------------------------------------------------
CONF_FILE_NAME = "robotnik_arm.json"
FIXED_FINGER_ANGLES = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
BASE_NAV_TIMEOUT = 25.0
ARM_REACH_LIMIT = 0.85
STANDOFF_Z_POINTS = [0.1, 0.5, 1.1, 1.25]
STANDOFF_POINTS = [0.80, 0.65, 0.65, 0.35]
REACH_TOLERANCE = 0.01

# Index layout of q_mes / qd_mes (17 joints): wheels, arm, fingers
WHEELS = slice(0, 4)
ARM = slice(4, 11)
FINGERS = slice(11, 17)

# Base navigation phase as an integer label (the expert's hidden state)
PHASE_CODES = {"align": 0, "drive": 1, "orient": 2, "done": 3}

# Arm links at or beyond this joint number are checked for collisions. Joints 1-2 sit on the
# base mount and touch the chassis by design, so they are excluded.
FIRST_CHECKED_ARM_JOINT = 3


def _git_hash():
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                             cwd=os.path.dirname(os.path.abspath(__file__)), timeout=5)
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _classify_stop(message):
    """Map the template's stop message to a short outcome label."""
    if message is None:
        return "unknown"
    if message.startswith("Target reached"):
        return "reached"
    if message.startswith("Target not reachable"):
        return "unreachable"
    if message.startswith("Reached step cap"):
        return "step_cap"
    return "unknown"


def run_episode(target, start_pose, seed, out_path=None, max_steps=40000, log_hz=100,
                cmd_noise_std=0.0, noise_hold_s=0.3, nudge_when="always", kick_prob=0.0, kick_std=15.0,
                use_gui=False, time_limit_s=None, obs_noise=0.0, delay_steps=0, floor_friction=1.0,
                base_mass_scale=1.0, hold_s=0.0, hold_unreachable=False):
    """Run the expert once and (optionally) save the recording.

    Args:
        target:        (x, y, z) end-effector target in the world frame, metres.
        start_pose:    (x, y, yaw) starting base pose, metres and radians.
        seed:          integer seed; drives any randomness inside the episode (command noise).
        out_path:      where to write the .npz file; None = do not save.
        max_steps:     step cap (the template's ROBOENV_MAX_STEPS default is 40000 = 40 s).
        log_hz:        recording rate. The sim runs at 1 kHz; 100 Hz keeps every 10th step.
        cmd_noise_std: noise added to the EXECUTED command (for recovery data, week 3).
                       The recorded label is always the expert's clean command.
                       Given as a fraction of each channel's limit (30 Nm torque, wheel speed).
        noise_hold_s:  each nudge is held this long before a new one is drawn. Noise that changes
                       every 1 ms averages out and barely moves the robot; held nudges push it
                       genuinely off course, so the expert's corrections get recorded.
        nudge_when:    "always", or "reach" = only nudge once the base has parked and the arm is
                       reaching (more examples of fine corrections near the target).
        kick_prob:     per-step chance (at 10 ms steps, while reaching) of a brief 50 ms torque kick that sets
                       the arm MOVING, so the expert's recovery from unexpected motion gets recorded.
        kick_std:      size of the kick, N*m (clipped by the torque limit).
        use_gui:       open the PyBullet window (slow; for watching one run).
        time_limit_s:  wall-clock safety limit; the run is stopped and marked "timeout".

        Domain randomisation (Stage B). The expert still controls from the TRUE state, so its
        commands stay perfect labels; what changes is the world, and what gets RECORDED as inputs:
        obs_noise:     std of sensor noise (same recipe as tier 4 in rollout.py). The recorded inputs
                       (base_pos, base_bearing, q_mes, qd_mes, ee_pos, tau_gravity) are the noisy
                       readings the model will see; the true ones are kept as *_true.
        delay_steps:   commands reach the motors this many 1-ms steps late.
        floor_friction: floor friction coefficient (template: 1.0).
        base_mass_scale: the base's mass is multiplied by this (heavier / lighter robot).
        hold_s:        keep running (and recording) the expert this long AFTER it reaches the target, so the
                       data shows what to do once there: hold still. The template stops the instant it
                       arrives, so without this the model never sees a single "arrived, stay put" example.

    Returns:
        dict with the summary ("meta") and the recorded arrays.
    """
    target = np.asarray(target, dtype=float)
    start_pose = np.asarray(start_pose, dtype=float)
    rng = np.random.default_rng(seed)
    noise_rng = np.random.default_rng([seed, 1])   # separate stream: adding noise must not change the nudges
    wall_start = time.time()

    # --------------------------------------------------------------------------------------
    # Simulation setup (as in the template)
    # --------------------------------------------------------------------------------------
    cur_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # configs/ and models/ are shared with the parent folder
    with quiet_native():  # hide the simulator's start-up warnings (set ROBOENV_VERBOSE=1 to see them)
        sim = pb.SimInterface(CONF_FILE_NAME, conf_file_path_ext=cur_dir, use_gui=use_gui)
    client = sim.pybullet_client
    try:
        sim.SetFloorFriction(float(floor_friction))
        rid = sim.bot[0].bot_pybullet
        if base_mass_scale != 1.0:
            base_mass = client.getDynamicsInfo(rid, -1)[0]
            client.changeDynamics(rid, -1, mass=base_mass * float(base_mass_scale))
        joint_names = [client.getJointInfo(rid, j)[1].decode() for j in range(client.getNumJoints(rid))]
        wheel_joint_ids = [j for j, n in enumerate(joint_names) if "wheel" in n]
        for j in wheel_joint_ids:
            client.changeDynamics(rid, j, contactStiffness=30000, contactDamping=1000,
                                  anisotropicFriction=[1.0, 0.01, 1.0])
        sim.bot[0].servo_motor_model.set_motor_gains(400, 30)

        # ----------------------------------------------------------------------------------
        # Random starting pose: change the start pose the simulator resets to, then reset.
        # Using the simulator's own ResetPose keeps its internal state consistent.
        # ----------------------------------------------------------------------------------
        z0 = float(sim.bot[0].link_floating_base_pos[2])
        quat = client.getQuaternionFromEuler([0.0, 0.0, float(start_pose[2])])
        sim.bot[0].link_floating_base_pos = [float(start_pose[0]), float(start_pose[1]), z0]
        sim.bot[0].link_floating_base_ori = list(quat)
        sim.ResetPose()
        sim.ReceiveObservation()

        target_finger_angles = configure_gripper_fingers(sim, client, rid, FIXED_FINGER_ANGLES)

        if use_gui:
            vis = client.createVisualShape(p.GEOM_SPHERE, radius=0.03, rgbaColor=[1.0, 0.0, 0.0, 0.85])
            client.createMultiBody(baseMass=0, baseVisualShapeIndex=vis, basePosition=target.tolist())

        # Links used for the collision check: arm links 3-7, the end effector and the fingers.
        # The arm's mounting plate ("joint_base") and links 1-2 sit on the chassis by design.
        def _checked(name):
            if not name.startswith("robot_j2s7s300_joint_"):
                return False
            suffix = name[len("robot_j2s7s300_joint_"):]
            if suffix == "base":
                return False
            return not suffix.isdigit() or int(suffix) >= FIRST_CHECKED_ARM_JOINT
        checked_links = [j for j, n in enumerate(joint_names) if _checked(n)]
        # Chassis parts the arm must not hit (not the arm mount, not the wheels)
        base_links = [-1] + [j for j, n in enumerate(joint_names)
                             if n in ("robot_base_footprint_joint", "robot_base_docking_joint")]
        ground = sim.ground_body

        # Pinocchio model
        ext_names = np.expand_dims(np.array(sim.getNameActiveJoints()), axis=0)
        with quiet_native():
            dyn_model = PinWrapper(CONF_FILE_NAME, "pybullet", ext_names, ["pybullet"], False, 0, cur_dir)
        ee_frame_id = dyn_model.pin_model.getFrameId("robot_j2s7s300_end_effector")
        shoulder_frame_id = dyn_model.pin_model.getFrameId("robot_j2s7s300_link_2")

        init_joint_angles = np.array(sim.GetMotorAngles(0))
        lower_limits, upper_limits = sim.GetBotJointsLimit()
        dt = sim.GetTimeStep()
        log_every = max(1, int(round(1.0 / (log_hz * dt))))

        # ----------------------------------------------------------------------------------
        # Expert plan (as in the template). Note: the base goal is placed relative to the
        # WORLD ORIGIN's view of the target, not relative to the robot's start.
        # ----------------------------------------------------------------------------------
        standoff = float(np.interp(target[2], STANDOFF_Z_POINTS, STANDOFF_POINTS))
        target_xy = target[:2]
        target_bearing = np.arctan2(target_xy[1], target_xy[0]) if np.linalg.norm(target_xy) > 1e-4 else 0.0
        base_goal_pos = target_xy - standoff * np.array([np.cos(target_bearing), np.sin(target_bearing)])
        base_goal_bearing = target_bearing

        base_state = BaseNavState()
        base_params = BaseControllerParams(nav_timeout=BASE_NAV_TIMEOUT)
        arm_params = ArmControllerParams(reach_limit=ARM_REACH_LIMIT)
        q_init_arm = init_joint_angles[ARM].copy()
        q_des_arm = q_init_arm.copy()
        qd_des_arm = np.zeros(7)
        arm_lower = np.array(lower_limits[ARM])
        arm_upper = np.array(upper_limits[ARM])
        stop_criteria = StopCriteria(reach_tolerance=REACH_TOLERANCE, max_steps=max_steps)
        cmd = MotorCommands()
        control_modes = ["velocity"] * 4 + ["torque"] * 7 + ["torque"] * 6

        # Command limits used to scale optional noise
        wheel_limit = (base_params.max_lin_vel + base_params.max_ang_vel * base_params.wheel_base_width / 2) \
            / base_params.wheel_radius
        noise_scale = np.concatenate([np.full(4, wheel_limit), np.full(7, arm_params.max_torque)])

        log = {k: [] for k in [
            "t", "base_pos", "base_quat", "base_bearing", "q_mes", "qd_mes", "ee_pos", "target",
            "phase", "nav_done", "q_des_arm", "qd_des_arm", "tau_gravity",
            "torque_arm", "wheel_cmds", "torque_arm_sent", "wheel_cmds_sent",
            "contact_ground", "contact_self", "base_pos_true", "base_bearing_true", "ee_pos_true", "holding",
        ]}
        hold_until, hold_max_err, reach_time_first = None, 0.0, None
        from collections import deque
        delay_line = deque([np.zeros(11)] * int(delay_steps))

        hold_steps = max(1, int(round(noise_hold_s / dt)))
        nudge = np.zeros(11)
        kick, kick_left, kick_count = np.zeros(7), 0, 0
        current_time = 0.0
        step = 0
        nav_done_time = None
        stop_message = None
        outcome = None
        ee_dist_error = np.inf
        min_ee_error = np.inf

        # Main loop, one iteration per 1 ms simulation step (identical in effect to the template's):
        #   measure -> expert decides -> (optional nudge/kick) -> log -> apply -> stop check
        while True:
            # --- Measure -------------------------------------------------------------------
            base_pos = np.array(sim.GetBasePosition())
            base_ori = np.array(sim.GetBaseOrientation())
            base_bearing = quaternion2bearing(base_ori[3], base_ori[0], base_ori[1], base_ori[2])
            q_mes = np.array(sim.GetMotorAngles(0))
            qd_mes = np.array(sim.GetMotorVelocities(0))

            q_pin = dyn_model.ReoderJoints2PinVec(np.concatenate((base_pos, base_ori, q_mes)), "pos")
            pin.forwardKinematics(dyn_model.pin_model, dyn_model.pin_data, q_pin)
            pin.updateFramePlacements(dyn_model.pin_model, dyn_model.pin_data)
            ee_pos = dyn_model.pin_data.oMf[ee_frame_id].translation.copy()
            ee_dist_error = float(np.linalg.norm(target - ee_pos))
            min_ee_error = min(min_ee_error, ee_dist_error)

            # --- Expert (unchanged controllers) --------------------------------------------
            phase_before = "done" if base_state.done else base_state.phase
            wheel_cmds = base_controller(base_state, base_pos, base_bearing, base_goal_pos,
                                         base_goal_bearing, current_time, base_params)
            nav_done = base_state.done
            if nav_done and nav_done_time is None:
                nav_done_time = current_time
            torque_arm, q_des_arm, qd_des_arm = arm_controller(
                dyn_model, q_pin, q_mes[ARM], qd_mes[ARM], q_des_arm, q_init_arm, target, ee_pos,
                ee_frame_id, shoulder_frame_id, arm_lower, arm_upper, dt, arm_params,
                reach_active=nav_done,
            )

            # --- Optional nudge on the EXECUTED command. The label logged below stays the expert's
            #     clean command, so the model learns "what the expert does here", not the noise.
            sent = np.concatenate((wheel_cmds, torque_arm))
            if cmd_noise_std > 0 and (nudge_when == "always" or nav_done):
                if step % hold_steps == 0:
                    nudge = rng.normal(0.0, cmd_noise_std, size=11) * noise_scale
                sent = sent + nudge
                sent[4:] = np.clip(sent[4:], -arm_params.max_torque, arm_params.max_torque)

            # --- Optional velocity kick (reaching only): a short sharp torque pulse ----------
            if kick_prob > 0 and nav_done:
                if kick_left == 0 and step % 10 == 0 and rng.random() < kick_prob:
                    kick, kick_left, kick_count = rng.normal(0.0, kick_std, size=7), 50, kick_count + 1
                if kick_left > 0:
                    sent = sent.copy()
                    sent[4:] = np.clip(sent[4:] + kick, -arm_params.max_torque, arm_params.max_torque)
                    kick_left -= 1

            # --- Optional command delay: the motors get the command from delay_steps ms ago -----
            if delay_steps > 0:
                delay_line.append(sent)
                sent = delay_line.popleft()

            # --- Record BEFORE the stop check, so the last step is kept (the template logs after
            #     it and so drops the final step). Every log_every-th step = 100 Hz by default.
            if step % log_every == 0:
                # Sensor noise (Stage B): record what a noisy sensor would have shown. Same recipe as
                # rollout.py's tier 4 (positions / angles: std obs_noise; speeds: 10x that). Gravity is
                # computed at the NOISY pose, because that is what the model adds back when it drives.
                if obs_noise > 0:
                    bp_seen = base_pos.copy()
                    bp_seen[:2] += noise_rng.normal(0, obs_noise, 2)
                    yaw_seen = base_bearing + noise_rng.normal(0, obs_noise)
                    bq_seen = np.array(client.getQuaternionFromEuler([0.0, 0.0, yaw_seen]))
                    q_seen = q_mes + noise_rng.normal(0, obs_noise, q_mes.shape)
                    qd_seen = qd_mes + noise_rng.normal(0, 10 * obs_noise, qd_mes.shape)
                    q_pin_seen = dyn_model.ReoderJoints2PinVec(np.concatenate((bp_seen, bq_seen, q_seen)), "pos")
                    pin.forwardKinematics(dyn_model.pin_model, dyn_model.pin_data, q_pin_seen)
                    pin.updateFramePlacements(dyn_model.pin_model, dyn_model.pin_data)
                    ee_seen = dyn_model.pin_data.oMf[ee_frame_id].translation.copy()
                else:
                    bp_seen, bq_seen, yaw_seen, q_seen, qd_seen, ee_seen, q_pin_seen = (
                        base_pos, base_ori, base_bearing, q_mes, qd_mes, ee_pos, q_pin)
                tau_g = pin.computeGeneralizedGravity(dyn_model.pin_model, dyn_model.pin_data, q_pin_seen)[6:13]
                ground_hit = any(c[3] in checked_links for c in client.getContactPoints(bodyA=rid, bodyB=ground))
                self_hit = any(
                    len(client.getClosestPoints(rid, rid, 0.0, linkIndexA=a, linkIndexB=b)) > 0
                    for a in checked_links for b in base_links
                )
                log["t"].append(current_time)
                log["base_pos"].append(bp_seen)
                log["base_quat"].append(bq_seen)
                log["base_bearing"].append(yaw_seen)
                log["q_mes"].append(q_seen)
                log["qd_mes"].append(qd_seen)
                log["ee_pos"].append(ee_seen)
                log["base_pos_true"].append(base_pos)
                log["base_bearing_true"].append(base_bearing)
                log["ee_pos_true"].append(ee_pos)
                log["target"].append(target)
                log["phase"].append(PHASE_CODES[phase_before])
                log["nav_done"].append(nav_done)
                log["q_des_arm"].append(q_des_arm.copy())
                log["qd_des_arm"].append(qd_des_arm.copy())
                log["tau_gravity"].append(tau_g)
                log["torque_arm"].append(torque_arm)
                log["wheel_cmds"].append(wheel_cmds)
                log["torque_arm_sent"].append(sent[4:])
                log["wheel_cmds_sent"].append(sent[:4])
                log["contact_ground"].append(ground_hit)
                log["contact_self"].append(self_hit)
                log["holding"].append(hold_until is not None)
                if ground_hit or self_hit:
                    outcome = "crashed"
                    stop_message = "Arm collision with " + ("ground" if ground_hit else "base")
                    break

            # --- Apply ------------------------------------------------------------------------
            cmd.SetControlCmd(np.concatenate((sent, np.zeros(6))), control_modes)
            sim.Step(cmd, "torque")

            # --- Stop conditions (template) --------------------------------------------------
            if hold_until is not None:                  # after arrival: keep holding, then stop
                hold_max_err = max(hold_max_err, ee_dist_error)
                if current_time >= hold_until:
                    break
            else:
                stop_message = check_stop(stop_criteria, current_time, dt, ee_dist_error, nav_done)
                if stop_message:
                    outcome = _classify_stop(stop_message)
                    arrived = outcome == "reached" or (outcome == "unreachable" and ee_dist_error < REACH_TOLERANCE)
                    graceful = hold_unreachable and outcome == "unreachable"   # stuck, stopped calmly
                    if hold_s > 0 and (arrived or graceful):
                        reach_time_first = current_time
                        hold_until = current_time + hold_s
                    else:
                        break
            if time_limit_s and time.time() - wall_start > time_limit_s:
                outcome, stop_message = "timeout", f"Wall-clock limit {time_limit_s}s"
                break

            current_time += dt
            step += 1
    finally:
        client.disconnect()

    # The template checks "stopped improving" BEFORE "reached", so a run that is already within
    # tolerance can be labelled a stall. We judge success by the final error instead.
    template_outcome = outcome
    if outcome == "unreachable" and ee_dist_error < REACH_TOLERANCE:
        outcome = "reached"

    # ------------------------------------------------------------------------------------------
    # Package the recording: float32 arrays + a JSON summary ("meta"), one .npz per run.
    # Written to a temp file and renamed, so a half-written file never looks complete.
    # ------------------------------------------------------------------------------------------
    arrays = {}
    for k, v in log.items():
        a = np.asarray(v)
        if a.dtype == bool:
            arrays[k] = a
        elif k == "phase":
            arrays[k] = a.astype(np.int8)
        else:
            arrays[k] = a.astype(np.float32)

    meta = {
        "seed": int(seed),
        "target": target.tolist(),
        "start_pose": start_pose.tolist(),
        "base_goal_pos": base_goal_pos.tolist(),
        "base_goal_bearing": float(base_goal_bearing),
        "standoff": standoff,
        "outcome": outcome,
        "stop_message": stop_message,
        "template_outcome": template_outcome,
        "final_ee_error": ee_dist_error,
        "min_ee_error": float(min_ee_error),
        "reach_time": (reach_time_first if reach_time_first is not None else float(current_time)) if outcome == "reached" else None,
        "nav_done_time": nav_done_time,
        "nav_timed_out": bool(nav_done_time is not None and nav_done_time >= BASE_NAV_TIMEOUT - dt),
        "duration": float(current_time),
        "n_sim_steps": int(step),
        "n_logged": int(len(log["t"])),
        "dt": float(dt),
        "log_every": int(log_every),
        "cmd_noise_std": float(cmd_noise_std),
        "noise_hold_s": float(noise_hold_s) if cmd_noise_std > 0 else None,
        "nudge_when": nudge_when if cmd_noise_std > 0 else None,
        "hold_s": float(hold_s), "hold_max_err": float(hold_max_err) if hold_until is not None else None,
        "held": bool(hold_until is not None and hold_max_err < REACH_TOLERANCE),
        "first_reach_time": reach_time_first,
        "obs_noise": float(obs_noise), "delay_steps": int(delay_steps),
        "floor_friction": float(floor_friction), "base_mass_scale": float(base_mass_scale),
        "kick_prob": float(kick_prob), "kick_std": float(kick_std), "n_kicks": int(kick_count),
        "max_steps": int(max_steps),
        "wall_time_s": float(time.time() - wall_start),
        "code_version": _git_hash(),
        "python": platform.python_version(),
        "pybullet_api": int(p.getAPIVersion()) if hasattr(p, "getAPIVersion") else None,
    }

    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        tmp = out_path + ".tmp.npz"
        np.savez_compressed(tmp, meta=np.array(json.dumps(meta)), **arrays)
        os.replace(tmp, out_path)  # write-then-rename: a half-written file never looks finished

    return {"meta": meta, **arrays}


def _demo():
    """Quick single run with the template's own target, from the template's own start."""
    res = run_episode(target=[0.0, 1.1, 0.5], start_pose=[0.0, 0.0, 0.0], seed=0,
                      use_gui=os.environ.get("ROBOENV_HEADLESS", "1") not in ("1", "true", "yes"))
    m = res["meta"]
    print(f"Outcome: {m['outcome']} | final error {m['final_ee_error'] * 100:.2f} cm | "
          f"time {m['duration']:.2f} s | {m['n_logged']} rows recorded")
    print("Expected for the template scene: reached | about 0.21 cm | 8.00 s")


if __name__ == "__main__":
    from quiet import run_main
    run_main(_demo, os.path.dirname(os.path.abspath(__file__)))
