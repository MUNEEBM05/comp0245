"""Our own test scenarios: a robot and a target with objects in the way. Frozen once, like the benchmark.

    python navigation/obstacle_scenes.py                      # writes scenes/obstacles_val.json and obstacles_test.json

Scenario ids (the "tier" field, so evaluate.py --tiers selects them):
    10 clear      no objects: same geometry, the control for the others
    11 sparse     a few boxes and cylinders, at least one on the direct route
    12 cluttered  many objects
    13 gap        a wall with one gap
    14 blocked    one large object right on the direct route

Target and start stay inside the ranges the models know (target 1.5-2 m from the centre, start 0.8-2 m), but on
opposite sides, so the drive is long enough for objects to matter. All objects are 1.6 m tall, so they cross the lidar plane.
Every scene is checked: A* finds a route with the planner's safety margin, and the start, parking spot, target and
the arm's reach are clear. Seeds: training 200000+, validation 300000+, test 400000+ (see training_scene).
"""
from _paths import ROOT, find_result  # noqa: F401  (project folder; lets the code folders import each other)
import argparse
import json
import os

import numpy as np

from planner import ROBOT_RADIUS_M, Grid, path_length, plan_path, rasterise
from samplers import _parking_goal, _uniform_disc, sample_target

HERE = ROOT
SCENARIOS = {10: "clear", 11: "sparse", 12: "cluttered", 13: "gap", 14: "blocked"}
SEED_BLOCK = {"train": 200_000, "val": 300_000, "test": 400_000}
HEIGHT_M = 1.6
TARGET_R, TARGET_Z, START_R = (1.5, 2.0), (0.3, 1.0), (0.8, 2.0)
START_CLEARANCE_M = ROBOT_RADIUS_M + 0.15
ARM_CLEARANCE_M = 1.0                     # around the parking spot and the target: the arm works here


def _distance(o, xy):
    """Distance from a point to the object's surface (0 if inside)."""
    d = np.asarray(xy, float) - np.asarray(o["xy"], float)
    if o["shape"] == "cylinder":
        return max(0.0, float(np.linalg.norm(d)) - o["radius"])
    c, s = np.cos(o["yaw"]), np.sin(o["yaw"])
    u, v = abs(c * d[0] + s * d[1]) - o["half_extents"][0], abs(-s * d[0] + c * d[1]) - o["half_extents"][1]
    return float(np.hypot(max(u, 0.0), max(v, 0.0)))


def _random_object(rng, xy):
    if rng.random() < 0.5:
        return {"shape": "cylinder", "xy": [float(xy[0]), float(xy[1])], "radius": float(rng.uniform(0.12, 0.3)),
                "height": HEIGHT_M}
    return {"shape": "box", "xy": [float(xy[0]), float(xy[1])], "yaw": float(rng.uniform(-np.pi, np.pi)),
            "half_extents": [float(rng.uniform(0.12, 0.35)), float(rng.uniform(0.12, 0.35))], "height": HEIGHT_M}


def _on_route(rng, start, goal, spread):
    """A point along the start-to-goal line, with some sideways offset."""
    u = (goal - start) / np.linalg.norm(goal - start)
    n = np.array([-u[1], u[0]])
    return start + rng.uniform(0.2, 0.8) * (goal - start) + rng.uniform(-spread, spread) * n


def _clear_of_key_places(o, start, goal, target):
    return (_distance(o, start) > START_CLEARANCE_M and _distance(o, goal) > ARM_CLEARANCE_M
            and _distance(o, target[:2]) > ARM_CLEARANCE_M)


def _blocks_line(obstacles, start, goal, margin=0.5):
    """Does the straight drive hit an object? margin = half the robot body's width + a little."""
    g = Grid()
    rasterise(g, obstacles)
    blocked = g.inflated(margin)
    pts = start + np.linspace(0, 1, 200)[:, None] * (goal - start)
    return any(blocked[g.to_cell(p)] for p in pts)


def _objects(kind, rng, start, goal, target):
    u = (goal - start) / np.linalg.norm(goal - start)
    angle = float(np.arctan2(u[1], u[0]))
    if kind == 13:                            # wall across the route with one gap
        gap_w = float(rng.uniform(1.9, 2.3))
        centre = start + rng.uniform(0.4, 0.6) * (goal - start)
        n = np.array([-u[1], u[0]])
        offset = float(rng.choice([-1, 1]) * (gap_w / 2 + rng.uniform(0.5, 0.9)))   # the direct line hits the wall
        parts = []
        for side in (-1, 1):
            half = 2.5
            mid = centre + (offset + side * (gap_w / 2 + half)) * n
            parts.append({"shape": "box", "xy": [float(mid[0]), float(mid[1])], "yaw": angle,
                          "half_extents": [0.08, half], "height": HEIGHT_M})
        return parts
    if kind == 14:                            # one large object on the line
        centre = start + rng.uniform(0.4, 0.6) * (goal - start)
        return [{"shape": "box", "xy": [float(centre[0]), float(centre[1])], "yaw": angle,
                 "half_extents": [float(rng.uniform(0.3, 0.5)), float(rng.uniform(0.5, 0.9))], "height": HEIGHT_M}]
    count, spread = (3, 0.5) if kind == 11 else (8, 1.6)
    out, tries = [], 0
    while len(out) < count and tries < 300:
        tries += 1
        o = _random_object(rng, _on_route(rng, start, goal, spread))
        if _clear_of_key_places(o, start, goal, target):
            out.append(o)
    return out


def make_scene(kind, seed, index=0):
    """One checked scene, or None if this seed's layout breaks a rule (the caller tries the next seed)."""
    rng = np.random.default_rng(seed)
    target = sample_target(rng, r_range=TARGET_R, z_range=TARGET_Z)
    goal = _parking_goal(target)
    b = np.arctan2(target[1], target[0])
    side = b + np.pi + rng.uniform(-0.8, 0.8)
    r = np.sqrt(rng.uniform(START_R[0] ** 2, START_R[1] ** 2))
    start = np.array([r * np.cos(side), r * np.sin(side)])
    if np.linalg.norm(goal - start) < 2.0:
        return None
    obstacles = [] if kind == 10 else _objects(kind, rng, start, goal, target)
    if kind == 11 and len(obstacles) < 3 or kind == 12 and len(obstacles) < 6:
        return None
    if not all(_clear_of_key_places(o, start, goal, target) for o in obstacles):
        return None
    if kind != 10 and not _blocks_line(obstacles, start, goal):
        return None
    path = plan_path(rasterise(Grid(), obstacles), start, goal)
    if path is None:
        return None
    return {"tier": kind, "index": index, "seed": int(seed), "scenario": SCENARIOS[kind],
            "target": [float(v) for v in target],
            "start_pose": [float(start[0]), float(start[1]), float(rng.uniform(-np.pi, np.pi))],
            "disturbance": {}, "obstacles": obstacles, "shortest_path_m": round(path_length(path), 3)}


def make_set(block, per_scenario):
    scenes = []
    for kind in SCENARIOS:
        first = SEED_BLOCK[block] + 10_000 * (kind - 10)
        seed, found = first, 0
        while found < per_scenario:
            if seed >= first + 10_000:
                raise RuntimeError(f"{SCENARIOS[kind]}: too few valid layouts in its seed block")
            s = make_scene(kind, seed, found)
            seed += 1
            if s is not None:
                scenes.append(s)
                found += 1
    return scenes


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-scenario", type=int, default=30)
    a = ap.parse_args()
    os.makedirs(os.path.join(HERE, "scenes"), exist_ok=True)
    for block in ("val", "test"):
        path = os.path.join(HERE, "scenes", f"obstacles_{block}.json")
        if os.path.exists(path):
            print(f"{path} is frozen; not overwriting")
            continue
        scenes = make_set(block, a.per_scenario)
        with open(path, "w") as fh:
            json.dump({"description": f"Obstacle scenarios ({block}). Frozen, do not edit.",
                       "per_tier": a.per_scenario, "scenes": scenes}, fh, indent=1)
        print(f"wrote {path}: {len(scenes)} scenes")


if __name__ == "__main__":
    main()
