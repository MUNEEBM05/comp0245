"""Inputs of the learned wheel network. The same function builds them for training (whole recorded runs) and driving (one row).

    lidar (36)      closest reading in each 10-degree direction around the robot, metres
    waypoint (6)    next route point seen from the robot: ahead, left, distance, sin and cos of the heading error, last-point flag
    parking (5)     parking spot seen from the robot: ahead, left, distance; sin and cos of the final-heading error
    motion (6)      how the robot moved over the last 0.1 s and 0.5 s: ahead, left, turned

Deliberately NOT inputs: measured wheel speeds (the team's "copycat trap": 0/50 and 9/50, the model repeats its last command),
map coordinates, time. Version B ("--no-waypoint" in train_driver.py) also leaves out the waypoint group.
"""
from _paths import ROOT, find_result  # noqa: F401  (project folder; lets the code folders import each other)
import numpy as np

SECTORS = 36
LAGS_ROWS = (10, 50)            # 0.1 s and 0.5 s at 100 Hz
NAMES = ([f"lidar_{i * 10:03d}" for i in range(SECTORS)]
         + ["wp_ahead", "wp_left", "wp_dist", "wp_sin", "wp_cos", "wp_last"]
         + ["park_ahead", "park_left", "park_dist", "park_sin", "park_cos"]
         + ["moved_ahead_100ms", "moved_left_100ms", "turned_100ms", "moved_ahead_500ms", "moved_left_500ms", "turned_500ms"])
WAYPOINT_COLUMNS = list(range(SECTORS, SECTORS + 6))


def lidar_sectors(ranges):
    """Closest reading per direction. Rays run from -180 degrees, in equal groups."""
    return np.asarray(ranges).reshape(SECTORS, -1).min(axis=1)


def _seen_from(xy, yaw, point):
    d = np.asarray(point) - np.asarray(xy)
    c, s = np.cos(yaw), np.sin(yaw)
    return c * d[..., 0] + s * d[..., 1], -s * d[..., 0] + c * d[..., 1]


def build(sectors, centre, yaw, waypoint, last, goal_centre, goal_bearing, past_xy_1, past_yaw_1, past_xy_2, past_yaw_2, xy):
    """Every argument is one row or a whole run (rows first). centre = the rotation centre; xy = the base reference point."""
    yaw = np.asarray(yaw, float)
    wf, wl = _seen_from(centre, yaw, waypoint)
    wd = np.hypot(wf, wl)
    gf, gl = _seen_from(centre, yaw, goal_centre)
    he = goal_bearing - yaw
    m1 = _seen_from(past_xy_1, past_yaw_1, xy)
    m2 = _seen_from(past_xy_2, past_yaw_2, xy)
    t1, t2 = np.sin(yaw - past_yaw_1), np.sin(yaw - past_yaw_2)
    safe = np.where(wd > 1e-6, wd, 1.0)
    return np.column_stack([np.atleast_2d(sectors), wf, wl, wd, wl / safe, wf / safe, last, gf, gl, np.hypot(gf, gl),
                            np.sin(he), np.cos(he), m1[0], m1[1], t1, m2[0], m2[1], t2]).astype(np.float32)


def from_run(a):
    """Inputs for a whole recorded run."""
    n = len(a["t"])
    i1, i2 = np.maximum(np.arange(n) - LAGS_ROWS[0], 0), np.maximum(np.arange(n) - LAGS_ROWS[1], 0)
    return build(a["lidar"], a["centre"], a["yaw"], a["waypoint"], a["wp_last"], a["goal_centre"], a["goal_bearing"],
                 a["xy"][i1], a["yaw"][i1], a["xy"][i2], a["yaw"][i2], a["xy"])
