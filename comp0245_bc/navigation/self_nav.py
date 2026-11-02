"""Drive with no position from the simulator: pose comes from wheel encoders + gyro, in the frame the robot started in. Lidar and the
camera see the real world from where the robot really is, but the robot only knows where it THINKS it is.

    mode "odo"     target given once at the start, then dead reckoning only
    mode "cam"     as "odo", and on parking the camera looks at the target and replaces the drifted estimate
    mode "search"  target NOT given: turn on the spot until the camera finds it, then as "cam"

Hand position comes from joint angles + the estimated base pose, so the hand-to-target error stays good if the target was seen from
the same estimated pose: dead-reckoning drift cancels out, and what is left is the camera's error.
Only rollout.py's scoring uses the true pose. Lidar and camera use it only to render what they would see, and the first pose defines the frame.
"""
from _paths import ROOT, find_result  # noqa: F401  (project folder; lets the code folders import each other)
from collections import deque

import numpy as np

from camera import TargetCamera, add_target_marker
from localisation import Odometry
from sensors import Gyro, WheelEncoders

START_STILL_S = 2.0        # gyro calibration
SEARCH_W = 0.4             # rad/s
SEARCH_EVERY_S = 0.25      # one picture per this long: 0.1 rad between pictures
FRAMES_AT_PARK = 5
LOOK_WITHIN_M = 2.5        # while driving, look at the target when it should be this close
LOOK_EVERY_S = 0.5
SETTLE_LOOK_EVERY_S = 0.25
GOAL_MOVED_M = 0.05        # a new look moves the parking spot by more than this: use the new spot
REPARK_M = 0.10            # on parking, the spot turns out this far away: drive to it
MAX_REPARKS = 3
REACH_LOOKS = 5            # while the arm reaches: target = median of this many recent looks
SETTLE_S = 1.5             # after parking the arm waits this long, looking: the filter corrects itself when the wheels stop, which moves the pose estimate
FREEZE_HAND_M = 0.0        # stop looking when the hand is this close to the target. 0: keep looking, the camera's sphere fit copes with a partly hidden target and the base creeps under the arm (encoders cannot see that)
MIN_VISIBLE = 0.2          # ignore a look with fewer pixels than this share of the parked view: the hand has covered nearly all of the target
NO_ROUTE_RESET_S = 1.0     # no route for this long: drift has smeared the map, start it again from new scans


class RealPoseLidar:
    """The rays start from where the robot really is, whatever it thinks."""

    def __init__(self, lidar, world):
        self.lidar, self.w, self.angles = lidar, world, lidar.angles

    def scan(self, _xy, _yaw):
        pos, q = self.w.client.getBasePositionAndOrientation(self.w.rid)
        return self.lidar.scan(np.array(pos[:2]), self.w.client.getEulerFromQuaternion(q)[2])


class SelfNavPolicy:
    def __init__(self, inner, mode="cam", seed=0):
        self.inner, self.mode, self.seed = inner, mode, seed
        self.name = f"selfnav_{mode}_{inner.name}"

    def _true(self):
        pos, q = self.w.client.getBasePositionAndOrientation(self.w.rid)
        return np.array(pos), self.w.client.getEulerFromQuaternion(q)[2]

    def reset(self, world, target):
        self.w, self.true_target = world, np.asarray(target, float)
        self.enc, self.gyro = WheelEncoders(seed=self.seed), Gyro(world, seed=self.seed)
        self.k, self.every = 0, max(1, int(round(0.01 / world.dt)))
        pos, yaw = self._true()
        self.odo = Odometry(pos[0], pos[1], yaw)
        self.est = np.array([pos[0], pos[1], yaw])
        self.camera = None
        self.stats = {"seen_at_park": None, "pixels_at_park": 0, "drift_at_park_m": None, "yaw_err_at_park_deg": None, "start_target_err_m": None,
                      "park_target_err_m": None, "park_target_err_before_m": None, "found_after_s": None, "true_park_err_m": None}
        self.looked, self.started, self.next_look, self.searched = False, False, 0.0, 0.0
        self.next_drive_look, self.reparks, self.no_route_since, self.map_resets = 0.0, 0, None, 0
        self.stats.update(reparks=0, map_resets=0, looks_seen=0, reach_looks=0)
        self.recent, self.full_view_px, self.settle_until = deque(maxlen=REACH_LOOKS), 0, 0.0
        if self.mode != "odo":
            add_target_marker(world, target)
            self.camera = TargetCamera(world, seed=self.seed)
        if self.mode != "search":
            self._start(self.true_target)

    def _start(self, target):
        self.inner.reset(self.w, target)
        self.inner.lidar = RealPoseLidar(self.inner.lidar, self.w)
        self.started = True

    def _seen_obs(self, obs):
        c, w = self.w.client, self.w
        quat = np.array(c.getQuaternionFromEuler([0.0, 0.0, float(self.est[2])]))
        base = np.array([self.est[0], self.est[1], obs["base_pos"][2]])
        ee, q_pin = w.forward_kinematics(base, quat, obs["q_mes"])
        return {**obs, "base_pos": base, "base_quat": quat, "base_bearing": float(self.est[2]), "ee_pos": ee, "q_pin": q_pin}

    def _measure(self, frames=1):
        """Target in the estimated frame, seen from the real pose."""
        pos, yaw = self._true()
        got, px = [], 0
        for _ in range(frames):
            m, px = self.camera.measure(pos[:2], yaw)
            if m is not None:
                got.append(TargetCamera.to_world(m, self.est[:2], self.est[2]))
        return (np.median(got, axis=0) if got else None), px

    def _search(self, obs):
        t = obs["t"]
        if t >= START_STILL_S and t >= self.next_look:
            self.next_look = t + SEARCH_EVERY_S
            found, _ = self._measure()
            if found is not None:
                self.stats["found_after_s"] = t - START_STILL_S
                self.stats["start_target_err_m"] = float(np.linalg.norm(found[:2] - self.true_target[:2]))
                self._start(found)
                return None
        if t < START_STILL_S or t - START_STILL_S > 2 * np.pi / SEARCH_W:    # one whole turn found nothing: stop
            return np.zeros(4), self._hold_torque(obs)
        p = self._search_params
        lw, rw = (-SEARCH_W * p.wheel_base_width / 2) / p.wheel_radius, (SEARCH_W * p.wheel_base_width / 2) / p.wheel_radius
        return np.array([rw, lw, lw, rw]), self._hold_torque(obs)

    def _hold_torque(self, obs):
        """The arm stays stowed while searching: the expert's controller with no reach."""
        from rollout import ExpertPolicy
        if not hasattr(self, "_holder") or self._holder_world is not self.w:
            self._holder, self._holder_world = ExpertPolicy(), self.w
            self._holder.reset(self.w, np.array([1.0, 0.0, 0.5]))
        self._holder.state.done = False     # never reach: the dummy target only exists to reuse the expert's holding torque
        return self._holder.act(obs)[1]

    @property
    def _search_params(self):
        from rollout import ExpertPolicy
        if not hasattr(self, "_p"):
            e = ExpertPolicy()
            e.reset(self.w, np.array([1.0, 0.0, 0.5]))
            self._p = e.base_params
        return self._p

    def act(self, obs):
        if self.k % self.every == 0:
            self.est = self.odo.update(self.enc.read(obs["qd_mes"]), self.gyro.read(0.01), 0.01)
        self.k += 1
        seen = self._seen_obs(obs)
        if not self.started:
            out = self._search(seen)
            if out is not None:
                return out
        wheels, torque = None, None
        inner = self.inner
        # Look just before the arm is told to reach: the parked check is about to pass or has just passed, the view is still clear
        wheels, torque = inner.act(seen)
        self._keep_map_usable(obs["t"])
        if self.camera is not None and inner.navigating:
            self._look_while_driving(obs["t"])
        if inner.expert.state.done and not self.looked:
            self.looked = True
            self._at_park(obs)
            if inner.navigating:               # re-parking: this step's reach must not start
                wheels = np.zeros(4)
            torque = inner._expert_arm(seen, hold_arm=True)[1]
        elif self.camera is not None and inner.expert.state.done:
            if obs["t"] < self.settle_until:         # keep the arm stowed; for model R too, whose own reach would start now
                torque = inner._expert_arm(seen, hold_arm=True)[1]
            self._look_while_reaching(obs["t"], seen)
        if obs["t"] < START_STILL_S:
            wheels = np.zeros(4)
        return wheels, torque

    def _keep_map_usable(self, t):
        from nav import LidarMap
        if not self.inner.navigating or self.inner.path is not None:
            self.no_route_since = None
        elif self.no_route_since is None:
            self.no_route_since = t
        elif t - self.no_route_since > NO_ROUTE_RESET_S:
            self.inner.map, self.no_route_since = LidarMap(), None
            self.map_resets += 1
            self.stats["map_resets"] = self.map_resets

    def _use_target(self, found):
        """New target estimate: the arm aims at it, and the parking spot follows from it by the same formula as at the start."""
        inner = self.inner
        keep_state, keep_hold = inner.expert.state, (inner.expert.q_des, inner.expert.q_hold)
        inner.expert.reset(self.w, found)
        inner.expert.state = keep_state
        inner.expert.q_des, inner.expert.q_hold = keep_hold
        m = inner.model
        while m is not None:          # MERGE: the arm may be wrapped (e.g. Kalman filter): give every layer the new target
            if hasattr(m, "target"):
                m.target = np.asarray(found, float)
            m = getattr(m, "inner", None)
        moved = float(np.linalg.norm(inner.expert.goal - inner.goal))
        return moved

    def _set_goal(self):
        inner = self.inner
        inner.goal = inner.expert.goal
        inner.goal_centre = inner._centre(inner.goal, inner.expert.goal_bearing)

    def _look_while_driving(self, t):
        inner = self.inner
        if t < self.next_drive_look or np.linalg.norm(inner.expert.target[:2] - self.est[:2]) > LOOK_WITHIN_M:
            return
        self.next_drive_look = t + LOOK_EVERY_S
        found, _ = self._measure()
        if found is None:
            return
        self.stats["looks_seen"] += 1
        if self._use_target(found) > GOAL_MOVED_M:
            self._set_goal()
            inner.path = None                  # replanned next step, from where the robot now is

    def _look_while_reaching(self, t, seen):
        """The pose estimate keeps changing a little after parking (the filter corrects itself when the wheels stop, and the base slides under
        the arm), so one look is not enough: the target has to be kept in step with the pose estimate."""
        if t < self.next_drive_look or np.linalg.norm(self.inner.expert.target - seen["ee_pos"]) < FREEZE_HAND_M:
            return
        self.next_drive_look = t + (SETTLE_LOOK_EVERY_S if t < self.settle_until else LOOK_EVERY_S)
        found, px = self._measure()
        if found is None or px < MIN_VISIBLE * self.full_view_px:
            return
        self.recent.append(found)
        self.stats["reach_looks"] += 1
        self._use_target(np.median(self.recent, axis=0))

    def _repark(self):
        inner = self.inner
        self._set_goal()
        expert_state = type(inner.expert.state)()
        inner.expert.state = expert_state
        inner.navigating, inner.t_handover, inner.path = True, None, None
        inner.follower = type(inner.follower)()
        self.looked = False
        self.reparks += 1
        self.stats["reparks"] = self.reparks

    def _at_park(self, obs):
        pos, yaw = self._true()
        s, inner = self.stats, self.inner
        s["drift_at_park_m"] = float(np.linalg.norm(self.est[:2] - pos[:2]))
        s["yaw_err_at_park_deg"] = float(np.degrees(abs((self.est[2] - yaw + np.pi) % (2 * np.pi) - np.pi)))
        s["true_park_err_m"] = float(np.linalg.norm(inner.expert.goal - self.est[:2]))
        before = self._target_error(inner.expert.target, pos, yaw)
        s["park_target_err_before_m"] = before
        if self.camera is None:
            return
        found, px = self._measure(FRAMES_AT_PARK)
        s["seen_at_park"], s["pixels_at_park"] = found is not None, px
        self.full_view_px, self.next_drive_look, self.settle_until = px, obs["t"] + SETTLE_LOOK_EVERY_S, obs["t"] + SETTLE_S
        self.recent.clear()
        if found is not None:
            moved = self._use_target(found)
            s["park_target_err_m"] = self._target_error(found, pos, yaw)
            if np.linalg.norm(inner.expert.goal - self.est[:2]) > REPARK_M and self.reparks < MAX_REPARKS:
                self._repark()

    def _target_error(self, est_target, pos, yaw):
        """How wrong the target is, as the robot sees it: its position from the estimated pose vs the truth from the true pose."""
        def local(t, x, y, a):
            d = np.array([t[0] - x, t[1] - y])
            return np.array([np.cos(a) * d[0] + np.sin(a) * d[1], -np.sin(a) * d[0] + np.cos(a) * d[1], t[2]])
        return float(np.linalg.norm(local(est_target, *self.est) - local(self.true_target, pos[0], pos[1], yaw)))
