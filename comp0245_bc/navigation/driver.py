"""Drive with the learned wheel network. The classical planner keeps running: version A takes its next waypoint as an input; in both
versions it tracks when the robot has parked, which is what the expert's arm waits for. Its wheel commands are never used.
Version B needs no waypoint, so the only classical parts it relies on when driving are that parked check and the arm.

    make_policy("driver:trained/driver_A.npz")                       learned wheels + expert arm
    make_policy("driver:trained/driver_A.npz:trained/vR_ik.npz")     learned wheels + model R's arm
    make_policy("driverbrake:trained/driver_A.npz")                  as first, and the wheels stop once the parked check says parked
"""
from _paths import ROOT, find_result  # noqa: F401  (project folder; lets the code folders import each other)
import json
import os
from collections import deque

import numpy as np

from driver_features import LAGS_ROWS, build, lidar_sectors
from nav import PlannerPolicy


class DriverPolicy(PlannerPolicy):
    def __init__(self, driver_path, arm_model=None, brake=False):
        super().__init__(arm_model)
        self.brake = brake
        z = np.load(driver_path)
        self.net = ([z[f"W{i}"] for i in range(int(z["n_layers"]))], [z[f"b{i}"] for i in range(int(z["n_layers"]))])
        self.x_mean, self.x_std, self.y_mean, self.y_std = z["x_mean"], z["x_std"], z["y_mean"], z["y_std"]
        self.x_lo, self.x_hi = z["x_lo"], z["x_hi"]
        self.cols = json.loads(str(z["meta"]))["columns"]
        self.name = ("driverbrake_" if brake else "driver_") + os.path.splitext(os.path.basename(driver_path))[0] + ("_" + self.model.name if self.model else "_expert")

    def reset(self, world, target):
        super().reset(world, target)
        self.k, self.every, self.row = 0, max(1, int(round(0.01 / world.dt))), None
        self.past = deque(maxlen=LAGS_ROWS[1] + 1)

    def _inputs(self, obs):
        xy, yaw = obs["base_pos"][:2].astype(float), float(obs["base_bearing"])
        centre = self._centre(xy, yaw)
        path = self.path if self.path is not None else [centre, self.goal_centre]
        i = min(self.follower.next, len(path) - 1) if self.navigating else len(path) - 1
        if self.k % self.every == 0:
            self.past.append((xy.copy(), yaw))
        lag1, lag2 = (self.past[-1 - min(n, len(self.past) - 1)] for n in LAGS_ROWS)
        row = build(lidar_sectors(self.last_ranges), centre, np.array([yaw]), path[i], np.array([float(i == len(path) - 1)]), self.goal_centre,
                    self.expert.goal_bearing, lag1[0], np.array([lag1[1]]), lag2[0], np.array([lag2[1]]), xy)
        return row[:, self.cols]

    def act(self, obs):
        # The planner's step runs first: it updates the lidar, map, route and parked check, and gives the arm's torque
        _, torque = super().act(obs)
        h = (np.clip(self._inputs(obs), self.x_lo, self.x_hi) - self.x_mean) / self.x_std
        Ws, bs = self.net
        for j, (W, b) in enumerate(zip(Ws, bs)):
            h = h @ W + b
            if j < len(Ws) - 1:
                h = np.maximum(h, 0.0)
        self.k += 1
        if self.brake and self.expert.state.done:
            return np.zeros(4), torque
        return (h * self.y_std + self.y_mean)[0], torque
