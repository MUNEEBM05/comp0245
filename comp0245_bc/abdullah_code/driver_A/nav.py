"""Classical navigation: lidar map -> A* route -> path follower -> the template's parking controller.

Stage 2 of the autonomy plan. Uses the true base pose and target; the arm is the expert's or a trained model's.
    make_policy("planner")                       planner + expert arm
    make_policy("planner:trained/vR_ik.npz")     planner + model R's arm (its wheel outputs are ignored)
"""
import numpy as np

from planner import BODY_RADIUS_M, Grid, plan_path, route_blocked
from robotnik_arm_control import base_controller, wrap_angle
from sensors import Lidar

SCAN_EVERY_S = 0.1
REPLAN_EVERY_S = 0.5
CORNER_REACHED_M = 0.12
ROTATION_CENTRE_M = 0.14          # measured: turning on the spot rotates the base about a point 0.12-0.17 m ahead of its
                                  # reference point. Steering that point means a turn cannot move it off the route.
APPROACH_M = 0.5
AT_GOAL_M = 0.08                  # the expert's position tolerance: the template's controller then does the final turn
MAX_V, MIN_V = 0.25, 0.05         # m/s, the expert's driving speed limits
MAX_DRIVE_W, MAX_TURN_W = 0.15, 0.6   # rad/s: small steering corrections while driving; turning on the spot
TURN_FIRST_RAD, ALIGNED_RAD = 0.2, 0.05   # stop and turn beyond the first; drive again within the second (expert's tolerance)
OCCUPIED_AFTER_HITS = 2           # a cell counts as occupied after this many lidar hits (ignores stray noise)


class LidarMap:
    def __init__(self, centre=(0.0, 0.0)):
        self.grid = Grid(centre)
        self.hits = np.zeros_like(self.grid.occ, dtype=np.int32)
        self.changed = True

    def add_scan(self, base_xy, yaw, angles, ranges, hit):
        a = angles[hit] + yaw
        pts = np.column_stack([base_xy[0] + ranges[hit] * np.cos(a), base_xy[1] + ranges[hit] * np.sin(a)])
        cells = np.floor((pts - self.grid.origin) / self.grid.res).astype(int)
        ok = np.all((cells >= 0) & (cells < self.grid.n), axis=1)
        np.add.at(self.hits, (cells[ok, 0], cells[ok, 1]), 1)
        occ = self.hits >= OCCUPIED_AFTER_HITS
        if np.any(occ != self.grid.occ):
            self.grid.occ, self.changed = occ, True


class PathFollower:
    """Straight segments only: drive straight, stop, turn on the spot, drive straight. Curving on this skid-steer base
    builds up sideways drift that friction barely removes (measured: wheels stopped from a straight line brake within
    0.5 s, but after curving the base kept sliding at 0.2 m/s during a turn on the spot)."""

    def __init__(self):
        self.turning, self.path, self.next = True, None, 1

    def command(self, path, xy, yaw):
        if path is not self.path:                    # a new route: start from its first segment
            self.path, self.next = path, 1
        while self.next < len(path) - 1 and np.linalg.norm(path[self.next] - xy) < CORNER_REACHED_M:
            self.next += 1
        aim = path[self.next]
        dist = float(np.linalg.norm(aim - xy))
        err = wrap_angle(np.arctan2(aim[1] - xy[1], aim[0] - xy[0]) - yaw)
        if self.turning and abs(err) < ALIGNED_RAD:
            self.turning = False
        elif not self.turning and abs(err) > TURN_FIRST_RAD:
            self.turning = True
        if self.turning:
            return 0.0, float(np.clip(2.0 * err, -MAX_TURN_W, MAX_TURN_W))
        return float(np.clip(dist, MIN_V, MAX_V)), float(np.clip(1.5 * err, -MAX_DRIVE_W, MAX_DRIVE_W))


def wheel_speeds(v, w, params):
    lw = (v - w * params.wheel_base_width / 2) / params.wheel_radius
    rw = (v + w * params.wheel_base_width / 2) / params.wheel_radius
    return np.array([rw, lw, lw, rw])


class PlannerPolicy:
    def __init__(self, arm_model=None, start_still_s=0.0):
        self.start_still_s = start_still_s            # stand still this long first: the gyro calibrates its bias
        from rollout import ExpertPolicy, MLPPolicy
        self.expert = ExpertPolicy()
        self.model = MLPPolicy(arm_model) if arm_model else None
        self.name = "planner_" + (self.model.name if self.model else "expert")

    def reset(self, world, target):
        self.expert.reset(world, target)
        if self.model:
            self.model.reset(world, target)
        self.w, self.goal, self.params = world, self.expert.goal, self.expert.base_params
        self.goal_centre = self._centre(self.goal, self.expert.goal_bearing)
        self.lidar = Lidar(world)
        self.map = LidarMap()
        self.path, self.navigating, self.next_scan, self.next_plan, self.t_handover = None, True, 0.0, 0.0, None
        self.replans, self.no_route_steps = 0, 0
        self.follower = PathFollower()

    def _centre(self, xy, yaw):
        return np.asarray(xy) + ROTATION_CENTRE_M * np.array([np.cos(yaw), np.sin(yaw)])

    def _route(self, centre):
        """To a point straight behind the parking spot, then straight in, so the robot arrives facing the target
        like the expert. A large final turn would move it off the spot, and the arm then reaches from too close."""
        b = self.expert.goal_bearing
        before = self.goal_centre - APPROACH_M * np.array([np.cos(b), np.sin(b)])
        if np.linalg.norm(before - centre) > APPROACH_M:
            path = plan_path(self.map.grid, centre, before, goal_margin=BODY_RADIUS_M)
            if path is not None and not route_blocked(self.map.grid, [before, self.goal_centre], radius=BODY_RADIUS_M):
                return path + [self.goal_centre]
        return plan_path(self.map.grid, centre, self.goal_centre)

    def _sense_and_plan(self, xy, yaw, centre, t):
        if t >= self.next_scan:
            ranges, hit = self.lidar.scan(xy, yaw)
            self.last_ranges = ranges
            self.map.add_scan(xy, yaw, self.lidar.angles, ranges, hit)
            self.next_scan = t + SCAN_EVERY_S
        if t >= self.next_plan or self.path is None:
            self.next_plan = t + REPLAN_EVERY_S
            stale = self.path is not None and self.map.changed and route_blocked(
                self.map.grid, self.path, self.follower.next, skip_near=centre)
            self.map.changed = False
            if self.path is None or stale:
                self.path = self._route(centre)
                self.replans += 1

    def act(self, obs):
        xy, yaw, t = obs["base_pos"][:2], float(obs["base_bearing"]), obs["t"]
        centre = self._centre(xy, yaw)
        if self.navigating:
            self._sense_and_plan(xy, yaw, centre, t)
            to_goal = self.goal_centre - centre
            dist = float(np.linalg.norm(to_goal))
            passed = dist < 0.15 and np.cos(wrap_angle(np.arctan2(to_goal[1], to_goal[0]) - yaw)) < 0.0
            if dist < AT_GOAL_M or passed:
                self.navigating, self.t_handover = False, t
                self.expert.state.phase = "orient"
        if t < self.start_still_s:
            wheels = np.zeros(4)
        elif self.navigating:
            if self.path is None:
                self.no_route_steps += 1
                wheels = np.zeros(4)
            else:
                wheels = wheel_speeds(*self.follower.command(self.path, centre, yaw), self.params)
        else:
            # Time since handover: the template's navigation timeout should not count the detour
            wheels = base_controller(self.expert.state, obs["base_pos"], obs["base_bearing"], self.goal,
                                     self.expert.goal_bearing, t - self.t_handover, self.params)
        if self.model:
            _, torque = self.model.act(obs)
        else:
            _, torque = self._expert_arm(obs)
        return wheels, torque

    def _expert_arm(self, obs, hold_arm=False):
        from rollout import ARM, arm_controller
        e, w = self.expert, self.w
        torque, e.q_des, _ = arm_controller(
            w.dyn_model, obs["q_pin"], obs["q_mes"][ARM], obs["qd_mes"][ARM], e.q_des, e.q_hold, e.target, obs["ee_pos"],
            w.ee_frame_id, w.shoulder_frame_id, w.arm_lower, w.arm_upper, w.dt, e.arm_params, reach_active=e.state.done and not hold_arm)
        return None, torque
