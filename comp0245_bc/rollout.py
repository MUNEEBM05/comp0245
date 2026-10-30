"""Let a policy (the expert, or our trained model) drive the robot on one scene.

This is the real test of behaviour cloning. Every 1 ms:
    1. read the robot (base pose, joint angles and speeds) and compute the hand position,
    2. the policy turns those readings + the target into 4 wheel speeds and 7 arm torques,
    3. the simulator applies them and moves one step.
Nothing from the expert is consulted while the model drives. Small per-step mistakes can add up
over thousands of steps, so a model with a tiny prediction error offline can still fail here.

The world is set up exactly as in expert_episode.py (same template settings, same random-start
mechanism). Every policy has the same two-method interface:
    policy.reset(world, target)
    wheel_cmds, torque_arm = policy.act(obs)

Policies:
    ExpertPolicy   the template's controllers, unchanged: the ceiling to compare against
    MLPPolicy      a trained model (MLP, linear or random forest), run with NumPy / scikit-learn
    HybridPolicy   diagnostic: expert drives one half (arm or wheels), the model the other
Metrics returned by rollout(): outcome, success, final / closest hand-to-target distance, time,
path length, command jerk, decision time, and where the base ended up (used by analyse_failures.py).
"""
import json
import os
import time
from collections import deque

import numpy as np

from expert_episode import (  # also imports the simulator quietly
    ARM, ARM_REACH_LIMIT, BASE_NAV_TIMEOUT, CONF_FILE_NAME, FIRST_CHECKED_ARM_JOINT, FIXED_FINGER_ANGLES,
    REACH_TOLERANCE, STANDOFF_POINTS, STANDOFF_Z_POINTS, MotorCommands, PinWrapper, p, pb, pin,
)
from features import HISTORY_LAGS_S, INTEGRAL_TAUS_S, ErrorAverager, goal_features, history_features, make_features
from quiet import quiet_native
from robotnik_arm_control import (
    ArmControllerParams, BaseControllerParams, BaseNavState, arm_controller, base_controller,
    configure_gripper_fingers, quaternion2bearing,
)

HERE = os.path.dirname(os.path.abspath(__file__))
MAX_TORQUE = 30.0                      # N*m, the expert's own torque limit
MAX_WHEEL = (0.25 + 1.0 * 0.66 / 2) / 0.11   # rad/s, fastest wheel speed the expert ever commands


# ------------------------------------------------------------------------------------------
# The simulated world
# ------------------------------------------------------------------------------------------
class World:
    """Everything a policy might need: the simulator, the robot model, limits and time step."""

    def __init__(self, start_pose, target, use_gui=False):
        cur_dir = HERE
        with quiet_native():
            self.sim = pb.SimInterface(CONF_FILE_NAME, conf_file_path_ext=cur_dir, use_gui=use_gui)
        sim = self.sim
        self.client = c = sim.pybullet_client
        sim.SetFloorFriction(1.0)
        self.rid = rid = sim.bot[0].bot_pybullet
        names = [c.getJointInfo(rid, j)[1].decode() for j in range(c.getNumJoints(rid))]
        for j, n in enumerate(names):
            if "wheel" in n:
                c.changeDynamics(rid, j, contactStiffness=30000, contactDamping=1000,
                                 anisotropicFriction=[1.0, 0.01, 1.0])
        sim.bot[0].servo_motor_model.set_motor_gains(400, 30)

        # Random start (same mechanism as expert_episode.py)
        z0 = float(sim.bot[0].link_floating_base_pos[2])
        quat = c.getQuaternionFromEuler([0.0, 0.0, float(start_pose[2])])
        sim.bot[0].link_floating_base_pos = [float(start_pose[0]), float(start_pose[1]), z0]
        sim.bot[0].link_floating_base_ori = list(quat)
        sim.ResetPose()
        sim.ReceiveObservation()
        configure_gripper_fingers(sim, c, rid, FIXED_FINGER_ANGLES)

        if use_gui:
            vis = c.createVisualShape(p.GEOM_SPHERE, radius=0.03, rgbaColor=[1.0, 0.0, 0.0, 0.85])
            c.createMultiBody(baseMass=0, baseVisualShapeIndex=vis, basePosition=list(map(float, target)))

        def checked(name):
            if not name.startswith("robot_j2s7s300_joint_"):
                return False
            suffix = name[len("robot_j2s7s300_joint_"):]
            return suffix != "base" and (not suffix.isdigit() or int(suffix) >= FIRST_CHECKED_ARM_JOINT)
        self.checked_links = [j for j, n in enumerate(names) if checked(n)]
        self.base_links = [-1] + [j for j, n in enumerate(names)
                                  if n in ("robot_base_footprint_joint", "robot_base_docking_joint")]
        self.ground = sim.ground_body

        ext = np.expand_dims(np.array(sim.getNameActiveJoints()), axis=0)
        with quiet_native():
            self.dyn_model = PinWrapper(CONF_FILE_NAME, "pybullet", ext, ["pybullet"], False, 0, cur_dir)
        self.ee_frame_id = self.dyn_model.pin_model.getFrameId("robot_j2s7s300_end_effector")
        self.shoulder_frame_id = self.dyn_model.pin_model.getFrameId("robot_j2s7s300_link_2")
        self.init_joint_angles = np.array(sim.GetMotorAngles(0))
        lo, hi = sim.GetBotJointsLimit()
        self.arm_lower, self.arm_upper = np.array(lo[ARM]), np.array(hi[ARM])
        self.dt = sim.GetTimeStep()
        self.cmd = MotorCommands()
        self.control_modes = ["velocity"] * 4 + ["torque"] * 7 + ["torque"] * 6

    def forward_kinematics(self, base_pos, base_quat, q_mes):
        """Hand position from measurements; leaves the robot model updated for this state."""
        q_pin = self.dyn_model.ReoderJoints2PinVec(np.concatenate((base_pos, base_quat, q_mes)), "pos")
        pin.forwardKinematics(self.dyn_model.pin_model, self.dyn_model.pin_data, q_pin)
        pin.updateFramePlacements(self.dyn_model.pin_model, self.dyn_model.pin_data)
        return self.dyn_model.pin_data.oMf[self.ee_frame_id].translation.copy(), q_pin

    def arm_collision(self):
        c, rid = self.client, self.rid
        if any(pt[3] in self.checked_links for pt in c.getContactPoints(bodyA=rid, bodyB=self.ground)):
            return "ground"
        for a in self.checked_links:
            for b in self.base_links:
                if c.getClosestPoints(rid, rid, 0.0, linkIndexA=a, linkIndexB=b):
                    return "base"
        return None

    def close(self):
        self.client.disconnect()


# ------------------------------------------------------------------------------------------
# Policies
# ------------------------------------------------------------------------------------------
class ExpertPolicy:
    """The template's controllers, unchanged: the ceiling every learned model is compared with."""
    name = "expert"

    def reset(self, world, target):
        self.w, self.target = world, np.asarray(target, float)
        standoff = float(np.interp(target[2], STANDOFF_Z_POINTS, STANDOFF_POINTS))
        b = np.arctan2(target[1], target[0]) if np.linalg.norm(target[:2]) > 1e-4 else 0.0
        self.goal = self.target[:2] - standoff * np.array([np.cos(b), np.sin(b)])
        self.goal_bearing = b
        self.state = BaseNavState()
        self.base_params = BaseControllerParams(nav_timeout=BASE_NAV_TIMEOUT)
        self.arm_params = ArmControllerParams(reach_limit=ARM_REACH_LIMIT)
        self.q_hold = world.init_joint_angles[ARM].copy()
        self.q_des = self.q_hold.copy()

    def act(self, obs):
        w = self.w
        wheels = base_controller(self.state, obs["base_pos"], obs["base_bearing"], self.goal,
                                 self.goal_bearing, obs["t"], self.base_params)
        torque, self.q_des, _ = arm_controller(
            w.dyn_model, obs["q_pin"], obs["q_mes"][ARM], obs["qd_mes"][ARM], self.q_des, self.q_hold,
            self.target, obs["ee_pos"], w.ee_frame_id, w.shoulder_frame_id, w.arm_lower, w.arm_upper,
            w.dt, self.arm_params, reach_active=self.state.done)
        return wheels, torque


class MLPPolicy:
    """A trained model driving the robot: an MLP or linear model (.npz, run with NumPy, no PyTorch
    needed) or a random forest (.pkl, scikit-learn).

    The model file carries everything needed to drive: the weights, the input/output scaling, the
    range of every input seen in training, and a "meta" record saying which optional input groups
    it was trained with (history, parking spot, integral, wheel speeds) and whether it predicts
    torque or torque-minus-gravity. act() rebuilds exactly the inputs train.py used.
    """

    def __init__(self, path):
        if path.endswith(".pkl"):
            import pickle
            with open(path, "rb") as fh:
                d = pickle.load(fh)
            self.forest = d["forest"]
            self.name = os.path.splitext(os.path.basename(path))[0]
            self.nets = []
            self.x_mean, self.x_std, self.y_mean, self.y_std = d["x_mean"], d["x_std"], d["y_mean"], d["y_std"]
            self.x_lo, self.x_hi = d.get("x_lo"), d.get("x_hi")
            self.clip_inputs = True
            self._read_meta(d["meta"])
            return
        self.forest = None
        z = np.load(path)
        self.name = os.path.splitext(os.path.basename(path))[0]
        meta0 = json.loads(str(z["meta"])) if "meta" in z.files else {}
        self.nets = []                                   # one network, or arm + wheels
        # Optional: which input columns each network sees (train_dagger.py --wheel-taus); default all
        self.cols = [meta0.get("group_inputs", {}).get(g) for g in meta0.get("groups", ["all"])]
        for g in meta0.get("groups", ["all"]):
            pre = "" if g == "all" else f"{g}_"
            n = int(z[f"{pre}n_layers"])
            self.nets.append(([z[f"{pre}W{i}"] for i in range(n)], [z[f"{pre}b{i}"] for i in range(n)]))
        self.x_mean, self.x_std = z["x_mean"], z["x_std"]
        self.y_mean, self.y_std = z["y_mean"], z["y_std"]
        self.x_lo = z["x_lo"] if "x_lo" in z.files else None
        self.x_hi = z["x_hi"] if "x_hi" in z.files else None
        self.clip_inputs = True   # anti-windup: keep inputs inside the range seen in training
        self._read_meta(json.loads(str(z["meta"])) if "meta" in z.files else {})

    def _read_meta(self, meta):
        """Which optional inputs / output convention this model was trained with (see train.py)."""
        self.minus_gravity = meta.get("torque") == "minus_gravity"
        self.history = meta.get("history_features", "none") == "deltas"
        self.goal = bool(meta.get("goal_features", False))
        self.integral = bool(meta.get("integral_features", False))
        self.integral_taus = tuple(meta.get("integral_taus") or INTEGRAL_TAUS_S)   # older models: 0.5 s, 2 s
        self.wheel_vel = bool(meta.get("wheel_vel_features", False))
        self.action = meta.get("action", "torque")
        self.arm_params = ArmControllerParams(reach_limit=ARM_REACH_LIMIT)   # same Kp, Kd as the expert

    def reset(self, world, target):
        """Called once per scene: remember the target and clear the short-term memories."""
        self.target = np.asarray(target, float)
        self.w = world
        # The training data was logged every 10 ms, so the history and integral inputs are
        # updated at that rate too (the control loop itself runs every 1 ms).
        self.sample_every = max(1, int(round(0.01 / world.dt)))
        self.buf = deque(maxlen=int(round(max(HISTORY_LAGS_S) / 0.01)) + 1)
        self.k = 0
        self.averager, self.avg_now = ErrorAverager(0.01, self.integral_taus), None

    def _history(self, obs):
        """History inputs: how the base and arm moved over the last 0.1 s and 0.5 s."""
        if self.k % self.sample_every == 0:
            self.buf.append((obs["base_pos"].copy(), float(obs["base_bearing"]), obs["q_mes"].copy()))
        lags = [min(int(round(l / 0.01)), len(self.buf) - 1) for l in HISTORY_LAGS_S]
        past = [self.buf[-1 - n] for n in lags]
        return history_features(obs["base_pos"], obs["base_bearing"], obs["q_mes"],
                                [p[0] for p in past], [p[1] for p in past], past[0][2])

    def predict(self, X):
        """Raw model output in real units (Nm and rad/s) for one row of inputs."""
        if self.clip_inputs and self.x_lo is not None:
            # Never feed the network a value outside the range it was trained on
            X = np.clip(X, self.x_lo, self.x_hi)
        x = (X - self.x_mean) / self.x_std          # same standardisation as training
        if self.forest is not None:
            return self.forest.predict(x) * self.y_std + self.y_mean
        outs = []
        for (Ws, bs), cols in zip(self.nets, self.cols):
            h = x if cols is None else x[:, cols]
            for i, (W, b) in enumerate(zip(Ws, bs)):
                h = h @ W + b
                if i < len(Ws) - 1:
                    h = np.maximum(h, 0.0)             # ReLU between layers, none on the output
            outs.append(h)
        return np.column_stack(outs) * self.y_std + self.y_mean

    def act(self, obs):
        """One control step: measurements + target -> (wheel speeds, arm torques)."""
        # 1. Build the input row, group by group, in the same order as train.py
        X = make_features(obs["base_pos"], obs["base_bearing"], obs["q_mes"], obs["qd_mes"],
                          obs["ee_pos"], self.target)
        if self.history:
            X = np.column_stack([X, self._history(obs)])
        if self.goal:
            X = np.column_stack([X, goal_features(obs["base_pos"], obs["base_bearing"], self.target)])
        if self.integral:
            if self.k % self.sample_every == 0:          # update every 10 ms, like the training data
                self.avg_now = self.averager.update(self.target - obs["ee_pos"])
            X = np.column_stack([X, self.avg_now[None, :]])
        if self.wheel_vel:
            X = np.column_stack([X, obs["qd_mes"][None, 0:4]])
        self.k += 1

        # 2. Ask the model
        y = self.predict(X)[0]

        # 3. Turn the model's output into motor commands
        m = self.w.dyn_model
        if self.action == "reference":
            # The model said where the arm should go; the template's PD law + gravity makes the torque
            p = self.arm_params
            torque = p.kp * y[:7] + p.kd * (y[7:14] - obs["qd_mes"][ARM])
            torque = torque + pin.computeGeneralizedGravity(m.pin_model, m.pin_data, obs["q_pin"])[6:13]
            y = np.concatenate([torque, y[14:]])
        else:
            torque = y[:7]
            if self.minus_gravity:  # the network predicted only the part left over after gravity
                torque = torque + pin.computeGeneralizedGravity(m.pin_model, m.pin_data, obs["q_pin"])[6:13]
        # 4. Same physical limits as the expert (the motors could not do more anyway)
        torque = np.clip(torque, -MAX_TORQUE, MAX_TORQUE)
        wheels = np.clip(y[7:], -MAX_WHEEL, MAX_WHEEL)
        return wheels, torque


def make_policy(spec):
    """'expert' or a path to a trained .npz file."""
    if spec == "expert":
        return ExpertPolicy()
    return MLPPolicy(spec)


# ------------------------------------------------------------------------------------------
# One drive
# ------------------------------------------------------------------------------------------
def rollout(policy, scene, max_time=40.0, use_gui=False, record_hz=100, seed=None):
    """Drive one scene with the given policy. Returns (metrics dict, trajectory dict at record_hz).

    A scene is a dict with "target", "start_pose" and "seed" (and optionally "disturbance" for
    tier 4: sensor noise, command delay and a sideways shove). The drive ends when the hand is
    within 1 cm of the target (success), on an arm collision, if the base tips over, if the model
    outputs NaN, or after max_time seconds.
    """
    target = np.asarray(scene["target"], float)
    dist = scene.get("disturbance") or {}
    rng = np.random.default_rng(scene["seed"] if seed is None else seed)
    world = World(scene["start_pose"], target, use_gui=use_gui)
    sim, c = world.sim, world.client
    try:
        policy.reset(world, target)
        dt = world.dt
        every = max(1, int(round(1.0 / (record_hz * dt))))
        delay = deque([np.zeros(11)] * int(dist.get("delay_steps", 0)))
        sigma = float(dist.get("obs_noise", 0.0))
        push_t, push_f = dist.get("push_time"), float(dist.get("push_force", 0.0))

        traj = {k: [] for k in ["t", "base_pos", "yaw", "ee_pos", "err", "wheel_cmds", "torque_arm"]}
        t, step, outcome, think = 0.0, 0, None, 0.0
        min_err, path_len, prev_xy, prev_cmd, jerk = np.inf, 0.0, None, None, []
        max_steps = int(round(max_time / dt))

        while True:
            base_pos = np.array(sim.GetBasePosition())
            base_quat = np.array(sim.GetBaseOrientation())
            bearing = quaternion2bearing(base_quat[3], base_quat[0], base_quat[1], base_quat[2])
            q_mes = np.array(sim.GetMotorAngles(0))
            qd_mes = np.array(sim.GetMotorVelocities(0))

            ee_true, q_pin = world.forward_kinematics(base_pos, base_quat, q_mes)
            err = float(np.linalg.norm(target - ee_true))
            min_err = min(min_err, err)

            # What the policy sees (with sensor noise on tier 4)
            if sigma > 0:
                bp = base_pos.copy()
                bp[:2] += rng.normal(0, sigma, 2)
                yaw_n = bearing + rng.normal(0, sigma)
                bq = np.array(c.getQuaternionFromEuler([0.0, 0.0, yaw_n]))
                qn = q_mes + rng.normal(0, sigma, q_mes.shape)
                qdn = qd_mes + rng.normal(0, 10 * sigma, qd_mes.shape)
                ee_seen, q_pin = world.forward_kinematics(bp, bq, qn)
                obs = {"t": t, "base_pos": bp, "base_quat": bq, "base_bearing": yaw_n,
                       "q_mes": qn, "qd_mes": qdn, "ee_pos": ee_seen, "q_pin": q_pin}
            else:
                obs = {"t": t, "base_pos": base_pos, "base_quat": base_quat, "base_bearing": bearing,
                       "q_mes": q_mes, "qd_mes": qd_mes, "ee_pos": ee_true, "q_pin": q_pin}

            tic = time.perf_counter()
            wheels, torque = policy.act(obs)
            think += time.perf_counter() - tic
            cmd = np.concatenate((wheels, torque))
            if not np.all(np.isfinite(cmd)):
                outcome = "unstable"
                break
            delay.append(cmd)
            applied = delay.popleft()

            if step % every == 0:
                traj["t"].append(t)
                traj["base_pos"].append(base_pos)
                traj["yaw"].append(bearing)
                traj["ee_pos"].append(ee_true)
                traj["err"].append(err)
                traj["wheel_cmds"].append(applied[:4])
                traj["torque_arm"].append(applied[4:])
                if prev_cmd is not None:
                    jerk.append(np.abs(applied - prev_cmd))
                prev_cmd = applied
                if prev_xy is not None:
                    path_len += float(np.linalg.norm(base_pos[:2] - prev_xy))
                prev_xy = base_pos[:2].copy()
                hit = world.arm_collision()
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

            # Shove the base sideways for 0.1 s (tier 4). The force is applied at the base link's
            # own origin in LINK_FRAME, so it pushes the robot rather than spinning it: a world-frame
            # point offset from the centre of mass would add a large torque.
            if push_t is not None and push_t <= t < push_t + 0.1:
                c.applyExternalForce(world.rid, -1, [0.0, push_f, 0.0], [0.0, 0.0, 0.0], p.LINK_FRAME)

            world.cmd.SetControlCmd(np.concatenate((applied, np.zeros(6))), world.control_modes)
            sim.Step(world.cmd, "torque")
            t += dt
            step += 1
    finally:
        world.close()

    jerk = np.array(jerk) if jerk else np.zeros((1, 11))

    # Where did it end up? (for sorting failures into reasons)
    standoff = float(np.interp(target[2], STANDOFF_Z_POINTS, STANDOFF_POINTS))
    tb = float(np.arctan2(target[1], target[0]))
    goal = target[:2] - standoff * np.array([np.cos(tb), np.sin(tb)])
    tr_bp = np.asarray(traj["base_pos"]) if traj["base_pos"] else np.zeros((1, 3))
    tr_yaw = np.unwrap(np.asarray(traj["yaw"], float)) if traj["yaw"] else np.zeros(1)
    last = max(1, int(round(2.0 * record_hz)))          # last 2 s
    yaw_rate = abs(tr_yaw[-1] - tr_yaw[-min(last, len(tr_yaw))]) / 2.0
    wheels_end = float(np.abs(np.asarray(traj["wheel_cmds"])[-last:]).mean()) if traj["wheel_cmds"] else 0.0
    face_err = float(np.degrees(abs((tb - tr_yaw[-1] + np.pi) % (2 * np.pi) - np.pi)))
    metrics = {
        "policy": policy.name, "tier": scene.get("tier"), "index": scene.get("index"), "seed": scene["seed"],
        "outcome": outcome, "success": outcome == "reached",
        "time_s": round(t, 3), "final_err_cm": round(err * 100, 2), "min_err_cm": round(min_err * 100, 2),
        "base_path_m": round(path_len, 3),
        "goal_dist_end_m": round(float(np.linalg.norm(tr_bp[-1, :2] - goal)), 3),
        "face_err_end_deg": round(face_err, 1),
        "yaw_rate_end": round(float(yaw_rate), 3),
        "wheels_end": round(wheels_end, 3),
        "torque_jerk": round(float(jerk[:, 4:].mean()), 4),   # mean change per 10 ms, Nm
        "wheel_jerk": round(float(jerk[:, :4].mean()), 4),    # mean change per 10 ms, rad/s
        "think_us_per_step": round(1e6 * think / max(step, 1), 1),
    }
    traj = {k: np.asarray(v, dtype=np.float32) for k, v in traj.items()}
    return metrics, traj


if __name__ == "__main__":
    # Sanity check: the expert, driven through this script, on the template's own scene.
    scene = {"seed": 0, "target": [0.0, 1.1, 0.5], "start_pose": [0.0, 0.0, 0.0]}
    m, _ = rollout(ExpertPolicy(), scene)
    print(json.dumps(m, indent=1))


class HybridPolicy:
    """Diagnostic: one half from the expert, the other half from a trained model.

    'mlp_arm'    -> expert drives the wheels, the model drives the arm
    'mlp_wheels' -> the model drives the wheels, the expert drives the arm
    Shows which half of the problem the model struggles with.
    """

    def __init__(self, path, mode):
        self.expert, self.mlp, self.mode = ExpertPolicy(), MLPPolicy(path), mode
        self.name = f"{self.mlp.name}_{mode}"

    def reset(self, world, target):
        self.expert.reset(world, target)
        self.mlp.reset(world, target)

    def act(self, obs):
        ew, et = self.expert.act(obs)
        mw, mt = self.mlp.act(obs)
        return (ew, mt) if self.mode == "mlp_arm" else (mw, et)


_make_policy_basic = make_policy


def make_policy(spec):  # noqa: F811  (extends the version above)
    """'expert', a path to a trained .npz, or 'path.npz:mlp_arm' / 'path.npz:mlp_wheels'."""
    if spec.endswith(":mlp_arm") or spec.endswith(":mlp_wheels"):
        path, mode = spec.rsplit(":", 1)
        return HybridPolicy(path, mode)
    return _make_policy_basic(spec)
