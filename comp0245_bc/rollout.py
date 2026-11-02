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
from features import HISTORY_LAGS_S, ErrorAverager, Smoother, goal_features, history_features, make_features
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
def add_objects(c, kind, start, target, seed):
    """Objects test (simple, all fixed in place, ready-made PyBullet objects from pybullet_data).

    kind 0: none (normal scene)
    kind 1: object AT the target - a 6 cm cube on a thin stand, its top 1.5 cm under the target point (the
            thing the hand reaches for). Touching it is allowed, it does not move.
    kind 2: clutter - 5 objects (table, ball, teddy, duck, lego) scattered around the room, but never within
            0.9 m of the robot's straight path to its parking spot, 0.8 m of the target or 0.7 m of the start.
            Touching one counts as a crash.
    kind 3: obstacle - one 0.3 x 0.3 x 0.5 m box halfway along the straight path to the parking spot
            (at least 0.75 m from the start).
            Touching it counts as a crash. Neither the expert nor the model can see it.
    kind 4: wall with a door - a 1 m tall wall across the straight path, halfway to the parking spot, with a
            1.4 m doorway 1.0 m to one side of the path (side chosen by the scene seed). The parking spot and
            target are hidden behind it. Touching the wall counts as a crash.
    Kinds 3 and 4 start the robot 2.5 m from its parking spot (same direction and heading as the scene),
    so the obstacle is always clearly in the way (see rollout()).
    The models' inputs never change: they still only get the robot's state and the target point.
    """
    import pybullet_data
    from samplers import _parking_goal
    c.setAdditionalSearchPath(pybullet_data.getDataPath())
    ids = {"target": [], "solid": []}
    goal = _parking_goal(target)
    if kind == 1:
        h = 0.03                                             # 6 cm cube on a thin stand (like an item on a shelf)
        top = target[2] - 0.015
        col = c.createCollisionShape(p.GEOM_BOX, halfExtents=[h, h, h])
        vis = c.createVisualShape(p.GEOM_BOX, halfExtents=[h, h, h], rgbaColor=[0.95, 0.6, 0.1, 1])
        ids["target"].append(c.createMultiBody(0, col, vis, [target[0], target[1], top - h]))
        pole = max(top - 2 * h, 0.01)
        col = c.createCollisionShape(p.GEOM_CYLINDER, radius=0.015, height=pole)
        vis = c.createVisualShape(p.GEOM_CYLINDER, radius=0.015, length=pole, rgbaColor=[0.4, 0.4, 0.4, 1])
        ids["target"].append(c.createMultiBody(0, col, vis, [target[0], target[1], pole / 2]))
    elif kind == 2:
        rng = np.random.default_rng(seed + 777)
        items = [("table/table.urdf", 0.6), ("soccerball.urdf", 0.35), ("teddy_vhacd.urdf", 3.0),
                 ("duck_vhacd.urdf", 3.0), ("lego/lego.urdf", 4.0)]
        a, b = start[:2], goal
        def far_enough(xy):
            ab = b - a
            u = np.clip(np.dot(xy - a, ab) / max(np.dot(ab, ab), 1e-9), 0, 1)
            return (np.linalg.norm(xy - (a + u * ab)) > 0.9 and np.linalg.norm(xy - target[:2]) > 0.8
                    and np.linalg.norm(xy - a) > 0.7)
        centre = (a + target[:2]) / 2
        for urdf, scale in items:
            for _ in range(200):
                xy = centre + rng.uniform(-2.5, 2.5, 2)
                if far_enough(xy):
                    yaw = rng.uniform(-np.pi, np.pi)
                    ids["solid"].append(c.loadURDF(urdf, [xy[0], xy[1], 0.0], c.getQuaternionFromEuler([0, 0, yaw]),
                                                   useFixedBase=True, globalScaling=scale))
                    break
    elif kind == 3:
        d = goal - start[:2]
        L = float(np.linalg.norm(d))
        mid = start[:2] + max(0.75, L / 2) * d / max(L, 1e-9)    # halfway, but never touching the robot at start
        he = [0.15, 0.15, 0.25]
        col = c.createCollisionShape(p.GEOM_BOX, halfExtents=he)
        vis = c.createVisualShape(p.GEOM_BOX, halfExtents=he, rgbaColor=[0.75, 0.15, 0.15, 1])
        ids["solid"].append(c.createMultiBody(0, col, vis, [mid[0], mid[1], he[2]]))
    elif kind == 4:
        d = goal - start[:2]
        L = float(np.linalg.norm(d))
        u = d / max(L, 1e-9)
        n = np.array([-u[1], u[0]])                                # sideways along the wall
        centre = start[:2] + (L / 2) * u
        side = 1.0 if (seed % 2 == 0) else -1.0
        door_c, door_w, half_len, h, th = side * 1.0, 1.4, 3.0, 1.0, 0.1
        yaw = float(np.arctan2(n[1], n[0]))
        for a0, a1 in ((-half_len, door_c - door_w / 2), (door_c + door_w / 2, half_len)):
            seg = a1 - a0
            he = [seg / 2, th / 2, h / 2]
            col = c.createCollisionShape(p.GEOM_BOX, halfExtents=he)
            vis = c.createVisualShape(p.GEOM_BOX, halfExtents=he, rgbaColor=[0.55, 0.55, 0.6, 1])
            xy = centre + ((a0 + a1) / 2) * n
            ids["solid"].append(c.createMultiBody(0, col, vis, [xy[0], xy[1], h / 2],
                                                  c.getQuaternionFromEuler([0, 0, yaw])))
    return ids


def _add_scene_obstacle(c, o):
    """MERGE (1 Nov), from Abdullah's rollout.World._add_obstacle: a fixed cylinder or box standing on the floor."""
    h = float(o["height"])
    if o["shape"] == "cylinder":
        shape = c.createCollisionShape(p.GEOM_CYLINDER, radius=float(o["radius"]), height=h)
        vis = c.createVisualShape(p.GEOM_CYLINDER, radius=float(o["radius"]), length=h, rgbaColor=[0.45, 0.5, 0.6, 1.0])
        quat = [0.0, 0.0, 0.0, 1.0]
    else:
        half = [float(o["half_extents"][0]), float(o["half_extents"][1]), h / 2]
        shape = c.createCollisionShape(p.GEOM_BOX, halfExtents=half)
        vis = c.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=[0.45, 0.5, 0.6, 1.0])
        quat = c.getQuaternionFromEuler([0.0, 0.0, float(o["yaw"])])
    return c.createMultiBody(baseMass=0, baseCollisionShapeIndex=shape, baseVisualShapeIndex=vis,
                             basePosition=[float(o["xy"][0]), float(o["xy"][1]), h / 2], baseOrientation=quat)


class World:
    """Everything a policy might need: the simulator, the robot model, limits and time step."""

    def __init__(self, start_pose, target, use_gui=False, floor_friction=1.0, base_mass_scale=1.0, payload_kg=0.0,
                 objects=0, objects_seed=0, obstacles=None):
        cur_dir = HERE
        with quiet_native():
            self.sim = pb.SimInterface(CONF_FILE_NAME, conf_file_path_ext=cur_dir, use_gui=use_gui)
        sim = self.sim
        self.client = c = sim.pybullet_client
        sim.SetFloorFriction(float(floor_friction))     # tier 5 changes it; the template uses 1.0
        self.rid = rid = sim.bot[0].bot_pybullet
        if base_mass_scale != 1.0:                      # tier 5: heavier / lighter base
            c.changeDynamics(rid, -1, mass=c.getDynamicsInfo(rid, -1)[0] * float(base_mass_scale))
        names = [c.getJointInfo(rid, j)[1].decode() for j in range(c.getNumJoints(rid))]
        if payload_kg > 0:   # Stage B: an object in the hand the robot model does not know about
            hand = names.index("robot_j2s7s300_joint_end_effector")
            c.changeDynamics(rid, hand, mass=c.getDynamicsInfo(rid, hand)[0] + float(payload_kg))
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

        self.object_ids = add_objects(c, int(objects), np.asarray(start_pose, float), np.asarray(target, float), objects_seed)
        if obstacles:   # MERGE (1 Nov): fixed obstacles from Abdullah's scene files; touching one counts as a crash, like our objects
            self.object_ids = dict(self.object_ids)
            self.object_ids["solid"] = list(self.object_ids.get("solid", [])) + [_add_scene_obstacle(c, o) for o in obstacles]

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
        for o in self.object_ids.get("solid", []):       # objects test: any part of the robot touching an object
            if c.getContactPoints(bodyA=rid, bodyB=o):
                return "object"
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
            self.pca_V = self.pca_s = None
            self._read_meta(d["meta"])
            return
        self.forest = None
        z = np.load(path)
        self.name = os.path.splitext(os.path.basename(path))[0]
        meta0 = json.loads(str(z["meta"])) if "meta" in z.files else {}
        self.nets = []                                   # one network, or arm + wheels
        for g in meta0.get("groups", ["all"]):
            pre = "" if g == "all" else f"{g}_"
            n = int(z[f"{pre}n_layers"])
            self.nets.append(([z[f"{pre}W{i}"] for i in range(n)], [z[f"{pre}b{i}"] for i in range(n)]))
        self.x_mean, self.x_std = z["x_mean"], z["x_std"]
        self.y_mean, self.y_std = z["y_mean"], z["y_std"]
        self.x_lo = z["x_lo"] if "x_lo" in z.files else None
        self.x_hi = z["x_hi"] if "x_hi" in z.files else None
        self.clip_inputs = True   # anti-windup: keep inputs inside the range seen in training
        self.pca_V = z["pca_V"] if "pca_V" in z.files else None
        self.pca_s = z["pca_s"] if "pca_s" in z.files else None
        self._read_meta(json.loads(str(z["meta"])) if "meta" in z.files else {})

    def _read_meta(self, meta):
        """Which optional inputs / output convention this model was trained with (see train.py)."""
        self.minus_gravity = meta.get("torque") == "minus_gravity"
        self.history = meta.get("history_features", "none") == "deltas"
        self.goal = bool(meta.get("goal_features", False))
        self.integral = bool(meta.get("integral_features", False))
        self.wheel_vel = bool(meta.get("wheel_vel_features", False))
        self.ik = bool(meta.get("ik_features", False))
        self.lidar = bool(meta.get("lidar_features", False))   # LiDAR slices (student of expert + navigation)
        self.relative = bool(meta.get("relative", False))   # absolute map positions dropped from the inputs
        self.drop_idx = []
        if meta.get("dropped"):                           # ablation models: inputs that were left out
            from features import ABSOLUTE_NAMES, feature_names_full
            names = feature_names_full(self.history, self.goal, self.integral, self.wheel_vel, self.ik)
            if self.relative:
                names = [n for n in names if n not in ABSOLUTE_NAMES]
            self.drop_idx = [i for i, n in enumerate(names) if any(n.startswith(p) for p in meta["dropped"])]
        self.action = meta.get("action", "torque")
        self.smooth = meta.get("smooth")    # model M: filter window in seconds (None for older models)
        self.smooth_qd = meta.get("smooth_qd", True)   # False = model M2: arm speeds stay raw (M has no entry)
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
        self.averager, self.avg_now = ErrorAverager(0.01), None
        self.lid_now = None
        if getattr(self, "ik", False):
            from ik_features import IKAverager
            self.ik_avg, self.ik_now = IKAverager(0.01), None
        self.smoother = Smoother(self.smooth, 0.01) if self.smooth else None
        self.sm_now = None

    def _history(self, base_pos, yaw, q):
        """History inputs: how the base and arm moved over the last 0.1 s and 0.5 s."""
        if self.k % self.sample_every == 0:
            self.buf.append((np.array(base_pos, float), float(yaw), np.array(q, float)))
        lags = [min(int(round(l / 0.01)), len(self.buf) - 1) for l in HISTORY_LAGS_S]
        past = [self.buf[-1 - n] for n in lags]
        return history_features(base_pos, yaw, q, [p[0] for p in past], [p[1] for p in past], past[0][2])

    def predict(self, X):
        """Raw model output in real units (Nm and rad/s) for one row of inputs."""
        if self.clip_inputs and self.x_lo is not None:
            # Never feed the network a value outside the range it was trained on
            X = np.clip(X, self.x_lo, self.x_hi)
        x = (X - self.x_mean) / self.x_std          # same standardisation as training
        if self.pca_V is not None:                  # PCA model: project onto the kept components
            x = (x @ self.pca_V) / self.pca_s
        if self.forest is not None:
            return self.forest.predict(x) * self.y_std + self.y_mean
        outs = []
        for Ws, bs in self.nets:
            h = x
            for i, (W, b) in enumerate(zip(Ws, bs)):
                h = h @ W + b
                if i < len(Ws) - 1:
                    h = np.maximum(h, 0.0)             # ReLU between layers, none on the output
            outs.append(h)
        return np.column_stack(outs) * self.y_std + self.y_mean

    def act(self, obs):
        """One control step: measurements + target -> (wheel speeds, arm torques)."""
        # 0. Model M: the arm speeds and history use readings smoothed every 10 ms, exactly as in
        #    training (features.smooth_run); every other input uses the raw readings.
        #    Model M2 smooths only the history: the arm speeds stay raw (they are the arm's brake).
        bp, yaw, q, qd = obs["base_pos"], obs["base_bearing"], obs["q_mes"], obs["qd_mes"]
        if self.smoother is not None:
            if self.k % self.sample_every == 0:
                self.sm_now = self.smoother.update(bp, yaw, q, qd)
            bp, yaw, q, qd = self.sm_now

        # 1. Build the input row, group by group, in the same order as train.py
        X = make_features(obs["base_pos"], obs["base_bearing"], obs["q_mes"],
                          qd if self.smooth_qd else obs["qd_mes"], obs["ee_pos"], self.target)
        if self.history:
            X = np.column_stack([X, self._history(bp, yaw, q)])
        if self.goal:
            X = np.column_stack([X, goal_features(obs["base_pos"], obs["base_bearing"], self.target)])
        if self.integral:
            if self.k % self.sample_every == 0:          # update every 10 ms, like the training data
                self.avg_now = self.averager.update(self.target - obs["ee_pos"])
            X = np.column_stack([X, self.avg_now[None, :]])
        if self.wheel_vel:
            X = np.column_stack([X, obs["qd_mes"][None, 0:4]])
        if self.ik:
            # fresh IK direction every step; its running averages advance every 10 ms like the recordings
            from ik_features import ik_rates
            w = self.w
            qd = ik_rates(w.dyn_model, w.ee_frame_id, w.shoulder_frame_id, obs["q_pin"], obs["ee_pos"],
                          self.target, self.arm_params)
            if self.k % self.sample_every == 0:
                self.ik_now = self.ik_avg.update(qd)
            X = np.column_stack([X, np.concatenate([qd, self.ik_now[7:]])[None, :].astype(np.float32)])
        if getattr(self, "lidar", False):
            # the model's own LiDAR scan, every 20 ms like the navigation layer whose drives it learned from
            if self.lid_now is None or self.k % LIDAR_EVERY == 0:
                from features import lidar_sectors
                d, ang = lidar_scan(self.w, obs["base_pos"][:2])
                self.lid_now = lidar_sectors(d, ang, float(obs["base_bearing"]))
            X = np.column_stack([X, self.lid_now[None, :].astype(np.float32)])
        if self.relative:
            from features import ABSOLUTE_IDX
            X = np.delete(X, ABSOLUTE_IDX, axis=1)
        if self.drop_idx:
            X = np.delete(X, self.drop_idx, axis=1)
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
def rollout(policy, scene, max_time=40.0, use_gui=False, record_hz=100, seed=None, hold_s=0.0):
    """Drive one scene with the given policy. Returns (metrics dict, trajectory dict at record_hz).

    A scene is a dict with "target", "start_pose" and "seed" (and optionally "disturbance" for
    tier 4: sensor noise, command delay and a sideways shove). The drive ends when the hand is
    within 1 cm of the target (success), on an arm collision, if the base tips over, if the model
    outputs NaN, or after max_time seconds.

    hold_s > 0: stricter success. The hand must STAY within 1 cm for hold_s seconds in a row (a real robot
    has to hold the pose to do anything there). Leaving the 1 cm zone restarts the clock. A drive that
    arrived but never held long enough ends as "reached_not_held".
    """
    target = np.asarray(scene["target"], float)
    dist = scene.get("disturbance") or {}
    if int(dist.get("objects", 0)) in (3, 4):      # obstacle tests: start 2.5 m from the parking spot, same direction
        from samplers import _parking_goal
        g, st = _parking_goal(target), np.asarray(scene["start_pose"], float)
        v = st[:2] - g
        v = v / np.linalg.norm(v) if np.linalg.norm(v) > 1e-6 else np.array([1.0, 0.0])
        scene = dict(scene, start_pose=[*(g + 2.5 * v), st[2]])
        max_time = max(max_time, float(os.environ.get("OBJ_TIME", "60")))  # a detour takes longer; same limit for every policy (OBJ_TIME only for the side check)
    if scene.get("obstacles"):                     # MERGE: Abdullah's obstacle scenes, same limit as our obstacle tests
        max_time = max(max_time, float(os.environ.get("OBJ_TIME", "60")))
    rng = np.random.default_rng(scene["seed"] if seed is None else seed)
    world = World(scene["start_pose"], target, use_gui=use_gui,
                  floor_friction=dist.get("floor_friction", 1.0), base_mass_scale=dist.get("base_mass_scale", 1.0),
                  payload_kg=dist.get("payload_kg", 0.0), objects=dist.get("objects", 0), objects_seed=scene["seed"],
                  obstacles=scene.get("obstacles"))
    sim, c = world.sim, world.client
    if hasattr(policy, "set_scene"):              # MERGE: camera mode seeds its sensor noise from the scene
        policy.set_scene(scene)
    try:
        policy.reset(world, target)
        dt = world.dt
        every = max(1, int(round(1.0 / (record_hz * dt))))
        delay = deque([np.zeros(11)] * int(dist.get("delay_steps", 0)))
        sigma = float(dist.get("obs_noise", 0.0))
        torque_sigma, tq_noise = float(dist.get("torque_noise", 0.0)), np.zeros(7)
        push_t, push_f = dist.get("push_time"), float(dist.get("push_force", 0.0))

        traj = {k: [] for k in ["t", "base_pos", "yaw", "ee_pos", "err", "wheel_cmds", "torque_arm"]}
        t, step, outcome, think = 0.0, 0, None, 0.0
        hold_start, first_reach = None, None
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
            # Motor noise (Stage B): the arm motors deliver the command +/- a random error, held 20 ms
            if torque_sigma > 0:
                if step % 20 == 0:
                    tq_noise = rng.normal(0.0, torque_sigma, 7)
                applied = applied.copy()
                applied[4:] = np.clip(applied[4:] + tq_noise, -MAX_TORQUE, MAX_TORQUE)

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
                if first_reach is None:
                    first_reach = t
                if hold_s <= 0:
                    outcome = "reached"
                    break
                hold_start = t if hold_start is None else hold_start
                if t - hold_start >= hold_s:
                    outcome = "reached"            # arrived AND held still for hold_s
                    break
            else:
                hold_start = None
            if step >= max_steps:
                outcome = "reached_not_held" if first_reach is not None else "timeout"
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
        "time_s": round(t, 3), "hold_s": hold_s,
        "first_reach_s": None if first_reach is None else round(first_reach, 3), "final_err_cm": round(err * 100, 2), "min_err_cm": round(min_err * 100, 2),
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
    if spec == "markov_expert":                     # memory-free expert used to label DAgger data
        from markov_expert import MarkovExpertPolicy
        return MarkovExpertPolicy()
    if spec.endswith(":mlp_arm") or spec.endswith(":mlp_wheels"):
        path, mode = spec.rsplit(":", 1)
        return HybridPolicy(path, mode)
    return _make_policy_basic(spec)


class GRUPolicy(MLPPolicy):
    """A trained GRU (train_gru.py) driving the robot, run with NumPy.

    Inputs are built exactly as for the MLPs (MLPPolicy.act). The memory h is advanced once per 10 ms,
    like the 100 Hz training data: row t uses h_{t-1}. In between, every 1 ms decision still uses the
    freshest measurements with that same h_{t-1}, so the arm's damping never works on stale readings.
    """

    def __init__(self, path):
        z = np.load(path)
        self.name = os.path.splitext(os.path.basename(path))[0]
        self.forest, self.nets = None, []
        for k in ("W_ih", "W_hh", "b_ih", "b_hh", "H_W0", "H_b0", "H_W1", "H_b1"):
            setattr(self, k, z[k])
        self.x_mean, self.x_std, self.y_mean, self.y_std = z["x_mean"], z["x_std"], z["y_mean"], z["y_std"]
        self.x_lo, self.x_hi, self.clip_inputs = z["x_lo"], z["x_hi"], True
        self._read_meta(json.loads(str(z["meta"])))
        self.n_h = self.W_hh.shape[1]

    def reset(self, world, target):
        super().reset(world, target)
        self.h_base = np.zeros(self.n_h, np.float32)     # h_{t-1}
        self.h_next = np.zeros(self.n_h, np.float32)     # h_t, becomes h_base at the next 10 ms row

    def _gru(self, x, h):
        gi = x @ self.W_ih.T + self.b_ih
        gh = h @ self.W_hh.T + self.b_hh
        n = self.n_h
        sig = lambda v: 1.0 / (1.0 + np.exp(-np.clip(v, -30.0, 30.0)))   # clipped: no overflow warnings
        r = sig(gi[:n] + gh[:n])
        zg = sig(gi[n:2 * n] + gh[n:2 * n])
        cand = np.tanh(gi[2 * n:] + r * gh[2 * n:])
        return (1.0 - zg) * cand + zg * h

    def predict(self, X):
        X = np.clip(X, self.x_lo, self.x_hi)
        x = ((X - self.x_mean) / self.x_std)[0].astype(np.float32)
        new_row = (self.k - 1) % self.sample_every == 0      # act() has already counted this step
        if new_row:
            self.h_base = self.h_next
        h = self._gru(x, self.h_base)
        if new_row:
            self.h_next = h
        y = np.maximum(h @ self.H_W0 + self.H_b0, 0.0) @ self.H_W1 + self.H_b1
        return (y * self.y_std + self.y_mean)[None, :]


class TCNPolicy(MLPPolicy):
    """A trained temporal convolution model (train_conv.py) driving the robot, run with NumPy.

    Each causal layer only needs its newest output: the outputs for past 10 ms rows are cached (they never
    change), so a decision costs about one small matrix product per layer. Like the GRU, the cache moves on
    once per 10 ms row, while every 1 ms decision uses the freshest measurements as the newest row.
    The model predicts the next `chunk` commands; the chunks predicted over the last rows that cover "now"
    are blended with weights exp(-k * age) (temporal ensembling), set by `ensemble_k` (None = no blending).
    """

    def __init__(self, path, ensemble_k="model"):
        z = np.load(path)
        self.name = os.path.splitext(os.path.basename(path))[0]
        self.forest, self.nets = None, []
        meta = json.loads(str(z["meta"]))
        self.kernel, self.dil, self.chunk = meta["kernel"], meta["dilations"], meta["chunk"]
        self.Ws = [z[f"C{i}_W"] for i in range(len(self.dil))]
        self.bs = [z[f"C{i}_b"] for i in range(len(self.dil))]
        self.H = [z[k] for k in ("H_W0", "H_b0", "H_W1", "H_b1", "H_W2", "H_b2") if k in z.files]
        self.skip = bool(meta.get("skip", False))
        self.x_mean, self.x_std, self.y_mean, self.y_std = z["x_mean"], z["x_std"], z["y_mean"], z["y_std"]
        self.x_lo, self.x_hi, self.clip_inputs = z["x_lo"], z["x_hi"], True
        self.k_ens = meta.get("ensemble_k") if ensemble_k == "model" else ensemble_k
        self._read_meta(meta)

    def reset(self, world, target):
        super().reset(world, target)
        self.bufs = None             # per layer: its INPUTS at past rows (newest last)
        self.chunks = deque(maxlen=self.chunk)

    def _layer(self, i, cur_in):
        """Newest output of layer i, from its newest input and the cached past inputs."""
        W, b, d, buf = self.Ws[i], self.bs[i], self.dil[i], self.bufs[i]
        out = b.copy()
        for j in range(self.kernel):
            back = (self.kernel - 1 - j) * d            # kernel tap j looks this many rows back
            v = cur_in if back == 0 else buf[-back]
            out += W[:, :, j] @ v
        return np.maximum(out, 0.0)

    def predict(self, X):
        X = np.clip(X, self.x_lo, self.x_hi)
        x = ((X - self.x_mean) / self.x_std)[0].astype(np.float32)
        new_row = (self.k - 1) % self.sample_every == 0
        if self.bufs is None:       # start of a drive: the past is the first reading repeated (as in training)
            self.bufs, v = [], x
            for i, d in enumerate(self.dil):
                self.bufs.append(deque([v] * ((self.kernel - 1) * d), maxlen=(self.kernel - 1) * d))
                v = self._layer(i, v)
        acts, v = [], x
        for i in range(len(self.dil)):
            acts.append(v)
            v = self._layer(i, v)
        h = np.concatenate([v, x]) if self.skip else v      # skip connection: the current row itself
        for j in range(0, len(self.H), 2):
            h = h @ self.H[j] + self.H[j + 1]
            if j < len(self.H) - 2:
                h = np.maximum(h, 0.0)
        pred = h.reshape(self.chunk, -1)
        # temporal ensembling: blend "now" from this prediction and from chunks made 1, 2, ... rows ago
        if self.k_ens is None or not self.chunks:
            y = pred[0]
        else:
            num, den = pred[0].copy(), 1.0
            for age, c in enumerate(reversed(self.chunks), start=1):
                if age < self.chunk:
                    w = float(np.exp(-self.k_ens * age))
                    num += w * c[age]
                    den += w
            y = num / den
        if new_row:                  # this row is now part of the past: cache it and its chunk
            for i in range(len(self.dil)):
                self.bufs[i].append(acts[i])
            self.chunks.append(pred)
        return (y * self.y_std + self.y_mean)[None, :]


class AttnPolicy(TCNPolicy):
    """A trained attention model (train_conv.py --arch attn): a small causal transformer over the last 61
    rows (0.61 s). Each moment attends to itself and older moments, so the results for the 60 past rows are
    computed once per 10 ms row and cached; every 1 ms decision only runs the newest moment through the
    blocks (attending to the cached keys and values). Chunk blending as in TCNPolicy."""

    def __init__(self, path, ensemble_k="model"):
        z = np.load(path)
        self.name = os.path.splitext(os.path.basename(path))[0]
        self.forest, self.nets = None, []
        meta = json.loads(str(z["meta"]))
        self.P = {k[3:].replace("__", "."): z[k] for k in z.files if k.startswith("A__")}
        self.n_blocks, self.heads, self.chunk = meta["blocks"], meta["heads"], meta["chunk"]
        self.T = self.P["pos"].shape[0]
        self.H = [z[k] for k in ("H_W0", "H_b0", "H_W1", "H_b1", "H_W2", "H_b2") if k in z.files]
        self.skip = bool(meta.get("skip", False))
        self.x_mean, self.x_std, self.y_mean, self.y_std = z["x_mean"], z["x_std"], z["y_mean"], z["y_std"]
        self.x_lo, self.x_hi, self.clip_inputs = z["x_lo"], z["x_hi"], True
        self.k_ens = meta.get("ensemble_k") if ensemble_k == "model" else ensemble_k
        self._read_meta(meta)

    def reset(self, world, target):
        MLPPolicy.reset(self, world, target)
        self.hist, self.cache, self.row_x = None, None, None
        self.chunks = deque(maxlen=self.chunk)

    def _lin(self, name, x):
        return x @ self.P[name + ".weight"].T + self.P[name + ".bias"]

    def _ln(self, name, x):
        m = x.mean(-1, keepdims=True)
        v = ((x - m) ** 2).mean(-1, keepdims=True)
        return (x - m) / np.sqrt(v + 1e-5) * self.P[name + ".weight"] + self.P[name + ".bias"]

    def _heads(self, x):            # (T, D) -> (heads, T, D/heads)
        return x.reshape(x.shape[0], self.heads, -1).transpose(1, 0, 2)

    def _past(self):
        """Run the 60 cached past rows through every block (causal), keeping each block's keys/values."""
        h = self._lin("emb", np.array(self.hist)) + self.P["pos"][:-1]
        n = h.shape[0]
        mask = np.triu(np.full((n, n), -1e9), 1)
        self.cache = []
        for i in range(self.n_blocks):
            b = f"blocks.{i}"
            zz = self._ln(b + ".ln1", h)
            q, k, v = (self._heads(self._lin(f"{b}.{w}", zz)) for w in "qkv")
            att = q @ k.transpose(0, 2, 1) / np.sqrt(q.shape[-1]) + mask
            att = np.exp(att - att.max(-1, keepdims=True))
            att /= att.sum(-1, keepdims=True)
            out = (att @ v).transpose(1, 0, 2).reshape(n, -1)
            self.cache.append((k, v))
            h = h + self._lin(b + ".o", out)
            h = h + self._lin(b + ".ff2", np.maximum(self._lin(b + ".ff1", self._ln(b + ".ln2", h)), 0.0))

    def predict(self, X):
        X = np.clip(X, self.x_lo, self.x_hi)
        x = ((X - self.x_mean) / self.x_std)[0].astype(np.float32)
        new_row = (self.k - 1) % self.sample_every == 0
        if self.hist is None:                       # start: the past is the first reading repeated
            self.hist = deque([x] * (self.T - 1), maxlen=self.T - 1)
            self._past()
        elif new_row:                               # last row's reading joins the past
            self.hist.append(self.row_x)
            self._past()
        if new_row:
            self.row_x = x
        h = self._lin("emb", x[None]) + self.P["pos"][-1]
        for i, (kc, vc) in enumerate(self.cache):
            b = f"blocks.{i}"
            zz = self._ln(b + ".ln1", h)
            q, k, v = (self._heads(self._lin(f"{b}.{w}", zz)) for w in "qkv")
            K, V = np.concatenate([kc, k], 1), np.concatenate([vc, v], 1)
            att = q @ K.transpose(0, 2, 1) / np.sqrt(q.shape[-1])
            att = np.exp(att - att.max(-1, keepdims=True))
            att /= att.sum(-1, keepdims=True)
            h = h + self._lin(b + ".o", (att @ V).transpose(1, 0, 2).reshape(1, -1))
            h = h + self._lin(b + ".ff2", np.maximum(self._lin(b + ".ff1", self._ln(b + ".ln2", h)), 0.0))
        zz = self._ln("lnf", h)[0]
        hh = np.concatenate([zz, x]) if self.skip else zz
        for j in range(0, len(self.H), 2):
            hh = hh @ self.H[j] + self.H[j + 1]
            if j < len(self.H) - 2:
                hh = np.maximum(hh, 0.0)
        pred = hh.reshape(self.chunk, -1)
        if self.k_ens is None or not self.chunks:
            y = pred[0]
        else:
            num, den = pred[0].copy(), 1.0
            for age, c in enumerate(reversed(self.chunks), start=1):
                if age < self.chunk:
                    w = float(np.exp(-self.k_ens * age))
                    num += w * c[age]
                    den += w
            y = num / den
        if new_row:
            self.chunks.append(pred)
        return (y * self.y_std + self.y_mean)[None, :]


_make_policy_mlp_or_hybrid = make_policy


def make_policy(spec):  # noqa: F811  (extends the versions above with GRU model files)
    if spec.endswith(".npz") and os.path.exists(spec):
        with np.load(spec) as z:
            kind = json.loads(str(z["meta"])).get("model") if "meta" in z.files else None
        if kind == "gru":
            return GRUPolicy(spec)
        if kind == "tcn":
            return TCNPolicy(spec)
        if kind == "attn":
            return AttnPolicy(spec)
    return _make_policy_mlp_or_hybrid(spec)


# ------------------------------------------------------------------------------------------
# Cancelling sensor noise (Stage B): a sensor filter and an ensemble, both wrapped around a trained model
# ------------------------------------------------------------------------------------------
class FilteredPolicy:
    """Smooths the readings before the model sees them: each sensor is averaged over the last `window_ms`
    milliseconds (moving average). Random noise that is fresh every millisecond mostly cancels out
    (averaging N readings divides it by about sqrt(N)); the real motion barely changes in that time.
    Cost: the readings lag by about window_ms / 2. The hand position and robot model are recomputed
    from the smoothed readings, so every input stays consistent. The model itself is unchanged."""

    def __init__(self, inner, window_ms, what="all"):
        self.inner, self.n, self.what = inner, int(window_ms), what      # what: all | base | noqd
        self.name = f"{inner.name}_filt{self.n}" + ("" if what == "all" else what)

    def reset(self, world, target):
        self.w = world
        self.buf = deque(maxlen=self.n)
        self.inner.reset(world, target)

    def act(self, obs):
        self.buf.append((obs["base_pos"].copy(), float(obs["base_bearing"]), obs["q_mes"].copy(), obs["qd_mes"].copy()))
        bp = np.mean([b[0] for b in self.buf], axis=0)
        yaws = np.unwrap([b[1] for b in self.buf])
        yaw = float((np.mean(yaws) + np.pi) % (2 * np.pi) - np.pi)
        q = np.mean([b[2] for b in self.buf], axis=0) if self.what in ("all", "noqd") else obs["q_mes"]
        qd = np.mean([b[3] for b in self.buf], axis=0) if self.what == "all" else obs["qd_mes"]
        bq = np.array(self.w.client.getQuaternionFromEuler([0.0, 0.0, yaw]))
        ee, q_pin = self.w.forward_kinematics(bp, bq, q)
        return self.inner.act({"t": obs["t"], "base_pos": bp, "base_quat": bq, "base_bearing": yaw,
                               "q_mes": q, "qd_mes": qd, "ee_pos": ee, "q_pin": q_pin})


class KalmanPolicy:
    """Kalman filter on the sensors before the model sees them (replaces the moving-average filter).

    One constant-velocity Kalman filter per channel: base x, base y, heading, and the 7 arm joints.
      predict  the channel keeps its current speed (state = [position, speed]; random acceleration qa).
      correct  blend the prediction with the noisy reading, weighted by how much each can be trusted.
    Arm joints fuse TWO sensors: the encoder angle and the joint speed sensor (sensor fusion), which is why
    they come out much cleaner than either reading alone. The base has no speed sensor, so position only.
    Noise levels are fixed from the sensor spec (0.005), the same for every test (not tuned per test).
    Unlike the moving average there is no fixed lag: the speed estimate lets it keep up with motion.
    vel=True also hands the model the filtered joint speeds (otherwise the raw ones, like filt20noqd)."""

    SIG, QA_ARM, QA_BASE = 0.005, 20.0, 10.0

    def __init__(self, inner, vel=False):
        self.inner, self.vel = inner, vel
        self.name = f"{inner.name}_kf" + ("v" if vel else "")

    def reset(self, world, target):
        self.w, self.s = world, None
        self.inner.reset(world, target)

    def act(self, obs):
        dt, sig = self.w.dt, self.SIG
        z = np.concatenate([obs["base_pos"][:2], [float(obs["base_bearing"])], obs["q_mes"]])
        n = len(z)
        qa = np.array([self.QA_BASE] * 3 + [self.QA_ARM] * (n - 3))
        if self.s is None:                                 # start at the first reading, speed from the sensor
            self.x, self.v = z.copy(), np.concatenate([np.zeros(3), obs["qd_mes"]])
            self.P = np.tile(np.diag([sig ** 2, 1.0]), (n, 1, 1))
            self.s = True
        else:
            z[2] = self.x[2] + _wrap(z[2] - self.x[2])     # heading: no jump at +-pi
            self.x = self.x + self.v * dt                  # predict
            P, a = self.P, qa ** 2
            P00 = P[:, 0, 0] + dt * (P[:, 0, 1] + P[:, 1, 0]) + dt * dt * P[:, 1, 1] + a * dt ** 4 / 4
            P01 = P[:, 0, 1] + dt * P[:, 1, 1] + a * dt ** 3 / 2
            P11 = P[:, 1, 1] + a * dt * dt
            self._upd(z, sig ** 2, 0, P00, P01, P11, np.ones(n, bool))
            zv = np.concatenate([np.zeros(3), obs["qd_mes"]])  # arm joints: also fuse the speed sensor
            self._upd(zv, (10 * sig) ** 2, 1, self.P[:, 0, 0], self.P[:, 0, 1], self.P[:, 1, 1],
                      np.arange(n) >= 3)
        bp = obs["base_pos"].copy(); bp[:2] = self.x[:2]
        yaw = float(_wrap(self.x[2]))
        q = self.x[3:].copy()
        qd = self.v[3:].copy() if self.vel else obs["qd_mes"]
        bq = np.array(self.w.client.getQuaternionFromEuler([0.0, 0.0, yaw]))
        ee, q_pin = self.w.forward_kinematics(bp, bq, q)
        return self.inner.act({"t": obs["t"], "base_pos": bp, "base_quat": bq, "base_bearing": yaw,
                               "q_mes": q, "qd_mes": qd, "ee_pos": ee, "q_pin": q_pin})

    def _upd(self, z, R, h, P00, P01, P11, mask):
        """Scalar Kalman update per channel, measuring position (h=0) or speed (h=1)."""
        if h == 0:
            S = P00 + R; K0, K1 = P00 / S, P01 / S; y = z - self.x
            n00, n01, n11 = (1 - K0) * P00, (1 - K0) * P01, P11 - K1 * P01
        else:
            S = P11 + R; K0, K1 = P01 / S, P11 / S; y = z - self.v
            n00, n01, n11 = P00 - K0 * P01, (1 - K1) * P01, (1 - K1) * P11
        m = mask
        self.x = np.where(m, self.x + K0 * y, self.x)
        self.v = np.where(m, self.v + K1 * y, self.v)
        P = np.empty((len(z), 2, 2))
        P[:, 0, 0] = np.where(m, n00, P00); P[:, 0, 1] = P[:, 1, 0] = np.where(m, n01, P01)
        P[:, 1, 1] = np.where(m, n11, P11)
        self.P = P


class EnsemblePolicy:
    """Averages the commands of several trained models (e.g. the 3 training runs of model R). Each model
    makes slightly different mistakes; averaging cancels part of them, including their reaction to noise."""

    def __init__(self, paths):
        self.members = [make_policy(p) for p in paths]
        self.name = "ens_" + "+".join(m.name for m in self.members)

    def reset(self, world, target):
        for m in self.members:
            m.reset(world, target)

    def act(self, obs):
        outs = [m.act(dict(obs)) for m in self.members]
        return np.mean([o[0] for o in outs], axis=0), np.mean([o[1] for o in outs], axis=0)


_make_policy_before_wrappers = make_policy


def make_policy(spec):  # noqa: F811  (adds 'ens:a.npz,b.npz,...' and a ':filt<ms>' suffix)
    if spec.endswith(":kf") or spec.endswith(":kfv"):
        inner, kind = spec.rsplit(":", 1)
        return KalmanPolicy(make_policy(inner), vel=(kind == "kfv"))
    if ":filt" in spec:
        inner, n = spec.rsplit(":filt", 1)
        what = "base" if n.endswith("base") else "noqd" if n.endswith("noqd") else "all"
        return FilteredPolicy(make_policy(inner), int(n.replace("base", "").replace("noqd", "")), what)
    if spec.startswith("ens:"):
        return EnsemblePolicy(spec[4:].split(","))
    return _make_policy_before_wrappers(spec)


# ------------------------------------------------------------------------------------------
# Going beyond the expert: LiDAR + VFH + Bug 2 navigation layer on top of a learned policy
# ------------------------------------------------------------------------------------------
def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class NavPolicy:
    """Classical obstacle avoidance on top of a policy (normally model R). The arm is never touched.

    LiDAR  a ring of 180 rays (every 2 deg, 4 m range) cast in PyBullet from the base at 30 cm height,
           50 times a second. Returns the distance to the nearest object in each direction.
    VFH    (Vector Field Histogram, Borenstein & Koren 1991) a direction is "blocked" if any LiDAR hit lies
           within the robot's safety radius (0.45 m) of that heading and closer than the look-ahead distance.
           The robot steers along the free direction closest to the goal, so it goes THROUGH a gap (door)
           when the gap is wide enough, instead of being pushed back like a potential field.
    Bug 2  when the straight line to the parking spot is blocked, the robot commits to one side (the side
           needing the smaller turn) and follows the obstacle edge on that side. It leaves the edge only when the
           parking spot is in clear view again AND it is closer than where it first hit the obstacle,
           so it cannot loop forever (the Bug 2 leave rule, using "goal visible" instead of "back on the line").
    Mode   while the way to the parking spot is clear, the learned policy drives everything (normal case is
           unchanged). The layer takes over the wheels only while the way is blocked; then hands back.
    """

    R_SAFE, LOOK, RANGE, N_RAYS, EVERY, ARM = 0.65, 1.5, 4.0, 180, 20, 0.9
    STUCK_FIX = os.environ.get("NAV_STUCK", "1") == "1"   # stuck detector on (default) / off (original layer)

    def __init__(self, inner):
        self.inner = inner
        self.name = f"{inner.name}_nav"

    def reset(self, world, target):
        from samplers import _parking_goal
        self.w, self.goal = world, _parking_goal(np.asarray(target, float))
        self.inner.reset(world, target)
        self.ang = np.linspace(-np.pi, np.pi, self.N_RAYS, endpoint=False)
        self.dist = np.full(self.N_RAYS, np.inf)
        self.mode, self.side, self.hit_d, self.step, self.wheels = "learned", 0.0, np.inf, 0, np.zeros(4)
        self.nav_steps, self.speed, self.prev_pos = 0, 0.0, np.zeros(2)
        self.final, self.orienting, self.phi, self.turning = False, False, None, False
        self.anchor, self.anchor_step, self.locked, self.lock_pos, self.n_stuck = None, 0, False, None, 0
        t = np.asarray(target, float)
        self.goal_bearing = float(np.arctan2(t[1], t[0]))
        self.arm_p = ArmControllerParams(reach_limit=ARM_REACH_LIMIT)

    def lidar(self, pos):
        c, z, r0 = self.w.client, 0.30, 0.40          # rays start just outside the robot body
        cs, sn = np.cos(self.ang), np.sin(self.ang)
        frm = [[pos[0] + r0 * a, pos[1] + r0 * b, z] for a, b in zip(cs, sn)]
        to = [[pos[0] + self.RANGE * a, pos[1] + self.RANGE * b, z] for a, b in zip(cs, sn)]
        out = c.rayTestBatch(frm, to)
        d = np.full(self.N_RAYS, np.inf)
        for i, h in enumerate(out):
            if h[0] >= 0 and h[0] != self.w.rid:
                d[i] = r0 + h[2] * (self.RANGE - r0)
        self.dist = d

    def blocked(self, phi, horizon, r=None):
        """Is the straight corridor of half-width R_SAFE, from the robot along heading phi for `horizon` metres,
        hit by any LiDAR point? (exact check for a round robot, so a door wider than 2 R_SAFE is seen as open)"""
        m = np.isfinite(self.dist)
        if not m.any():
            return False
        d, a = self.dist[m], self.ang[m] - phi
        along, lat = d * np.cos(a), np.abs(d * np.sin(a))
        return bool(np.any((along > -0.1) & (along < horizon) & (lat < (self.R_SAFE if r is None else r))))

    def steer_dir(self, pos, phi_g, d_goal):
        """Direction the local search starts from. Bug 2 / VFH: straight at the parking spot."""
        return phi_g

    def pick_side(self, phi_g):
        """Bug 2 needs a side. Pick the side with the open direction (LiDAR sees nothing for 3.5 m, e.g. a
        doorway or the end of the obstacle) closest to the goal direction."""
        open_ = ~(self.dist < 3.5)
        for k in range(0, 91):
            for sd in (1.0, -1.0):
                i = int(np.argmin(np.abs(_wrap(self.ang - (phi_g + sd * np.radians(2 * k))))))
                if open_[i]:
                    return sd
        return 1.0

    def act(self, obs):
        wheels, torque = self.inner.act(obs)              # learned policy always runs (keeps its history)
        pos, yaw = obs["base_pos"][:2], float(obs["base_bearing"])
        to_g = self.goal - pos
        d_goal = float(np.linalg.norm(to_g))
        phi_g = float(np.arctan2(to_g[1], to_g[0]))
        if self.step % self.EVERY == 0:
            self.speed = float(np.linalg.norm(pos - self.prev_pos)) / (self.EVERY * self.w.dt) if self.step else 0.0
            self.prev_pos = pos.copy()
            self.lidar(pos)
            clear = not self.blocked(phi_g, d_goal + 0.2)
            if self.mode == "learned" and not self.final:
                rel = np.abs(_wrap(self.ang - yaw))
                close_ahead = bool(np.any((self.dist < 0.8) & (rel < np.radians(60))))
                too_close = bool(np.min(self.dist) < 0.65)          # safety bubble: something right next to us
                if (not clear and (self.blocked(phi_g, min(d_goal + 0.2, 1.6)) or close_ahead)) or too_close:
                    self.mode, self.side = "nav", 0.0
            if self.mode == "nav":
                if self.side == 0.0:          # commit to one side (Bug 2 boundary following)
                    self.side = self.pick_side(phi_g)
                if (d_goal < 0.15 or self.orienting) and not self.orienting:
                    self.turning = False
                if d_goal < 0.15 or self.orienting:
                    # At the parking spot: face the target like the expert does, then hand everything back to the
                    # learned policy for the final parking and the reach.
                    self.orienting = True
                    err = _wrap(self.goal_bearing - yaw)
                    if abs(err) < 0.08:
                        v, w = 0.0, 0.0
                        self.mode, self.final = "learned", True
                    elif not self.turning and self.speed > 0.03:
                        v, w = 0.0, 0.0
                    else:
                        self.turning = True
                        v, w = 0.0, float(np.clip(1.5 * err, -0.4, 0.4))
                    lw, rw = (v - w * 0.33) / 0.11, (v + w * 0.33) / 0.11
                    self.wheels = np.clip(np.array([rw, lw, lw, rw]), -MAX_WHEEL, MAX_WHEEL)
            if self.mode == "nav" and not self.orienting:
                phi = None
                phi_g = self.steer_dir(pos, phi_g, d_goal)            # MapNavPolicy: towards the A* path instead
                for side in (self.side, -self.side):
                    for k in range(0, 91):
                        cand = phi_g + side * np.radians(2 * k)
                        if not self.blocked(cand, min(self.LOOK, d_goal + 0.2)):
                            phi = cand; break
                    if phi is not None:
                        break
                # Stuck detector: if the layer has not moved 0.15 m in 5 s, it is dithering between two headings
                # (one sits on the edge of the safety corridor and flickers free/blocked as the base turns). Then
                # lock the current heading until the robot has driven 0.5 m. Runs that make progress never trigger it.
                if not self.STUCK_FIX:
                    pass
                elif self.anchor is None or np.linalg.norm(pos - self.anchor) > 0.15:
                    self.anchor, self.anchor_step = pos.copy(), self.step
                elif not self.locked and self.phi is not None and (self.step - self.anchor_step) * self.w.dt > 5.0:
                    self.locked, self.lock_pos, self.n_stuck = True, pos.copy(), self.n_stuck + 1
                if self.locked and (np.linalg.norm(pos - self.lock_pos) > 0.5
                                    or self.blocked(self.phi, min(self.LOOK, d_goal + 0.2), self.R_SAFE - 0.1)):
                    self.locked = False
                    self.anchor, self.anchor_step = pos.copy(), self.step
                # keep the current heading while it stays free (avoids stop-start), unless one much closer to the
                # goal has opened up
                if not self.locked and phi is not None and (
                        self.phi is None or self.blocked(self.phi, min(self.LOOK, d_goal + 0.2))
                        or abs(_wrap(phi - phi_g)) < abs(_wrap(self.phi - phi_g)) - np.radians(15)):
                    self.phi = phi
                phi = self.phi
                if phi is None:                                       # boxed in: turn on the spot
                    v, w = 0.0, 0.25 * self.side
                else:
                    # The wheels barely grip sideways (the base glides like on ice once it moves sideways), so the
                    # layer never turns while moving: stop, turn on the spot, then drive dead straight.
                    # The arm sticks out ~0.75 m in front, so a turn is only made if nothing is within arm reach in
                    # the arc the front sweeps through; otherwise back up a little first.
                    err = _wrap(phi - yaw)
                    rel = _wrap(self.ang - yaw)
                    lo, hi = min(0.0, err) - 0.35, max(0.0, err) + 0.35
                    sweep = np.any((self.dist < self.ARM) & (rel > lo) & (rel < hi))
                    rear_free = not np.any((self.dist < 0.6) & (np.abs(rel) > np.radians(130)))
                    if self.turning and abs(err) < 0.05:
                        self.turning = False
                    if abs(err) > 0.1 or self.turning:
                        if not self.turning and self.speed > 0.03:      # stop first
                            v, w = 0.0, 0.0
                        elif not self.turning and sweep and rear_free:
                            v, w = -0.1, 0.0
                        else:                                           # turn on the spot until lined up
                            self.turning = True
                            v, w = 0.0, float(np.clip(1.5 * err, -0.4, 0.4))
                    else:
                        front = self.dist[np.abs(rel) < np.radians(30)]
                        fmin = np.min(front) if front.size else np.inf
                        v = (0.1 if fmin < 0.9 else 0.2 if fmin < 1.5 else 0.35) * min(1.0, max(0.25, d_goal / 0.6))
                        w = float(np.clip(1.0 * err, -0.05, 0.05))
                self.dbg = (phi, v, w)
                lw, rw = (v - w * 0.33) / 0.11, (v + w * 0.33) / 0.11
                self.wheels = np.clip(np.array([rw, lw, lw, rw]), -MAX_WHEEL, MAX_WHEEL)
        self.step += 1
        if self.mode == "nav":
            # While the layer drives, the arm is held folded in its start pose (PD + gravity, like the template's
            # arm controller). Out on a detour the learned arm would see situations it never trained on and swing,
            # and the base slides sideways easily. The learned policy takes the arm back with the wheels.
            self.nav_steps += 1
            m, p = self.w.dyn_model, self.arm_p
            g = pin.computeGeneralizedGravity(m.pin_model, m.pin_data, obs["q_pin"])[6:13]
            tq = p.kp * (self.w.init_joint_angles[ARM] - obs["q_mes"][ARM]) - p.kd * obs["qd_mes"][ARM] + g
            return self.wheels.copy(), np.clip(tq, -MAX_TORQUE, MAX_TORQUE)
        return wheels, torque


class MapNavPolicy(NavPolicy):
    """Same layer, plus a MAP and a GLOBAL PLANNER (the standard mobile-robot stack, e.g. ROS Nav2):

    Map     occupancy grid, 5 cm cells, built only from the robot's own LiDAR hits (nothing from the simulator's
            object list). Every cell any ray has hit stays occupied, so the robot REMEMBERS walls it has seen,
            including once they are behind it. Unseen cells count as free (optimistic, standard for unknown space).
    Inflate cells closer than the robot's radius (0.6 m) to an occupied cell are forbidden; cells within 1.2 m
            cost extra, so the path keeps to the middle of a gap (Nav2's "inflation layer").
    A*      8-connected shortest path from the robot to the parking spot on that grid, re-planned twice a second
            as the map fills in.
    Follow  the local layer (VFH corridor check + stop-turn-go) steers towards the farthest point of the path it
            can see in a straight, collision-free line (pure pursuit with line of sight), instead of straight at
            the goal. So behind a long wall it heads for the wall's END that the map remembers, not into it.
    """
    CELL, R_INFL, R_COST, REPLAN = 0.05, 0.60, 1.2, 500

    def __init__(self, inner):
        super().__init__(inner)
        self.name = f"{inner.name}_mapnav"

    def reset(self, world, target):
        super().reset(world, target)
        self.origin = None                       # grid placed on the first scan, around robot and goal
        self.occ, self.path, self.plan_step, self.trail, self.n_plans, self.free = None, None, -10 ** 9, [], 0, None

    def lidar(self, pos):
        """Every scan also goes into the map, from the very start (not only once the layer takes over)."""
        super().lidar(pos)
        if self.origin is None:
            self._grid_setup(pos)
        self._add_scan(pos)

    def _grid_setup(self, pos):
        lo = np.minimum(pos, self.goal) - 4.5
        hi = np.maximum(pos, self.goal) + 4.5
        self.origin = lo
        self.shape = tuple(np.ceil((hi - lo) / self.CELL).astype(int))
        self.occ = np.zeros(self.shape, bool)

    def _cell(self, p):
        ij = np.floor((np.asarray(p) - self.origin) / self.CELL).astype(int)
        return np.clip(ij, 0, np.array(self.shape) - 1)

    def _xy(self, ij):
        return self.origin + (np.asarray(ij) + 0.5) * self.CELL

    def _add_scan(self, pos):
        m = np.isfinite(self.dist)
        pts = pos + self.dist[m, None] * np.stack([np.cos(self.ang[m]), np.sin(self.ang[m])], 1)
        ij = np.floor((pts - self.origin) / self.CELL).astype(int)
        ok = (ij >= 0).all(1) & (ij < np.array(self.shape)).all(1)
        self.occ[ij[ok, 0], ij[ok, 1]] = True

    def _plan(self, pos):
        import heapq
        from scipy import ndimage
        dist = ndimage.distance_transform_edt(~self.occ) * self.CELL if self.occ.any() else np.full(self.shape, 9.0)
        free = dist >= self.R_INFL
        cost = 1.0 + 4.0 * np.clip((self.R_COST - dist) / (self.R_COST - self.R_INFL), 0, 1)
        s, g = tuple(self._cell(pos)), tuple(self._cell(self.goal))
        free[s] = free[g] = True                 # the robot may already be inside an inflated band
        H, W = self.shape
        gx, gy = g
        best = {s: 0.0}
        parent = {s: None}
        heap = [(0.0, s)]
        nbr = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
               (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414)]
        while heap:
            f, c = heapq.heappop(heap)
            if c == g:
                break
            gc = best[c]
            if f > gc + np.hypot(c[0] - gx, c[1] - gy) + 1e-9:
                continue                          # stale entry
            for dx, dy, l in nbr:
                n = (c[0] + dx, c[1] + dy)
                if not (0 <= n[0] < H and 0 <= n[1] < W) or not free[n]:
                    continue
                ng = gc + l * cost[n]
                if ng < best.get(n, np.inf):
                    best[n], parent[n] = ng, c
                    heapq.heappush(heap, (ng + np.hypot(n[0] - gx, n[1] - gy), n))
        self.free = free
        if g not in parent:
            return None                           # no path on the map: fall back to plain Bug 2
        out, c = [], g
        while c is not None:
            out.append(c)
            c = parent[c]
        return self._xy(np.array(out[::-1]))

    def _still_ok(self, path, pos):
        """Remaining part of the current path: still collision-free on the updated map? Returns its length or None."""
        if path is None or self.free is None:
            return None
        k0 = int(np.argmin(np.linalg.norm(path - pos, axis=1)))
        rest = path[k0:]
        ij = self._cell(rest)
        if not self.free[ij[:, 0], ij[:, 1]][1:-1].all():
            return None
        return float(np.linalg.norm(np.diff(rest, axis=0), axis=1).sum()) + float(np.linalg.norm(rest[0] - pos))

    def steer_dir(self, pos, phi_g, d_goal):
        if self.step - self.plan_step >= self.REPLAN or self.path is None:
            # Path hysteresis: two routes round an obstacle can be almost equally short, and re-planning would flip
            # between them (the robot turns left, right, left...). Keep the current path while it is still free,
            # unless the new one is clearly (15%) shorter.
            new, self.plan_step, self.n_plans = self._plan(pos), self.step, self.n_plans + 1
            keep = self._still_ok(self.path, pos)
            if new is None or keep is None:
                self.path = new if new is not None else (self.path if keep is not None else None)
            else:
                l_new = float(np.linalg.norm(np.diff(new, axis=0), axis=1).sum())
                if l_new < 0.85 * keep:
                    self.path = new
        self.trail.append(pos.copy())
        if self.path is None or len(self.path) < 2:
            return phi_g
        k0 = int(np.argmin(np.linalg.norm(self.path - pos, axis=1)))     # where on the path we are
        aim = self.path[min(k0 + 4, len(self.path) - 1)]
        for p in self.path[k0:][::-1]:                                     # farthest visible point, within 3 m
            v = p - pos
            dv = float(np.linalg.norm(v))
            if dv < 3.0 and (dv < 0.05 or not self.blocked(float(np.arctan2(v[1], v[0])), dv)):
                aim = p
                break
        v = aim - pos
        return float(np.arctan2(v[1], v[0])) if np.linalg.norm(v) > 0.05 else phi_g


_make_policy_before_nav = make_policy


def make_policy(spec):  # noqa: F811  (adds ':nav' (LiDAR + VFH + Bug 2) and ':mapnav' (+ map + A*) suffixes)
    if spec.endswith(":mapnav"):
        return MapNavPolicy(make_policy(spec[:-7]))
    if spec.endswith(":nav"):
        return NavPolicy(make_policy(spec[:-4]))
    return _make_policy_before_nav(spec)


# ------------------------------------------------------------------------------------------
# Shared LiDAR (same rays as the navigation layer): used by models trained with --lidar and by record_nav.py
# ------------------------------------------------------------------------------------------
LIDAR_EVERY = 20                                       # one scan every 20 ms (50 Hz)


def lidar_scan(world, pos, n_rays=180, max_range=4.0):
    """Ring of n_rays rays at 30 cm height, starting 0.40 m out from the base centre (just outside the body).
    Returns (distances, world-frame angles); inf where nothing is hit within max_range."""
    c, z, r0 = world.client, 0.30, 0.40
    ang = np.linspace(-np.pi, np.pi, n_rays, endpoint=False)
    cs, sn = np.cos(ang), np.sin(ang)
    frm = [[pos[0] + r0 * a, pos[1] + r0 * b, z] for a, b in zip(cs, sn)]
    to = [[pos[0] + max_range * a, pos[1] + max_range * b, z] for a, b in zip(cs, sn)]
    d = np.full(n_rays, np.inf)
    for i, h in enumerate(c.rayTestBatch(frm, to)):
        if h[0] >= 0 and h[0] != world.rid:
            d[i] = r0 + h[2] * (max_range - r0)
    return d, ang


# ------------------------------------------------------------------------------------------
# MERGE (1 Nov): Abdullah's navigation as policy specs. The arm can be any team spec.
#   planner                                           A* planner + expert arm
#   planner:trained/bo_t03.npz:kfv                    A* planner + BO arm with Kalman filter
#   driverbrake:trained/driver_A.npz:trained/bo_t03.npz:kfv   learned Driver A (+ brake) + BO arm with Kalman
_make_policy_before_merge = make_policy


def make_policy(spec):  # noqa: F811
    if spec == "planner" or spec.startswith("planner:"):
        from nav import PlannerPolicy
        return PlannerPolicy(spec.split(":", 1)[1] if ":" in spec else None)
    if spec.startswith(("driver:", "driverbrake:")):
        from driver import DriverPolicy
        parts = spec.split(":", 2)
        return DriverPolicy(parts[1], parts[2] if len(parts) > 2 else None, brake=parts[0] == "driverbrake")
    if spec.startswith("cam:"):      # Abdullah's self_nav "cam" mode: no true robot pose (encoders + gyro), camera fixes the target
        from self_nav import SelfNavPolicy

        class _Cam(SelfNavPolicy):
            def set_scene(self, scene):        # same sensor-noise seed as his eval_selfnav.py (results match it exactly)
                self.seed = scene["seed"] % 100_000
        return _Cam(make_policy(spec[4:]), "cam")
    return _make_policy_before_merge(spec)
