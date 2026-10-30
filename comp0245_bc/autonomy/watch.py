"""Watch one scene drive in the 3D viewer, with what the robot senses and plans drawn on top.

    export LIBGL_ALWAYS_SOFTWARE=1      # needed for the viewer on WSL at the moment
    python watch.py --policy planner --scenes scenes/obstacles_val.json --tier 14 --index 0 --max-time 90
    python watch.py --policy planner:trained/vR_ik.npz --scenes scenes/obstacles_val.json --tier 12 --index 3 --max-time 90 --top
    python watch.py --policy trained/vR_ik.npz --tier 1 --index 40          # a model on a benchmark scene

Drawn:  red dots    lidar hits
        blue line   the planned route (planner policies only)
        green line  the path the robot has actually driven
        grey dots   cells on the robot's map that count as occupied (planner policies only, with --show-map)

The drawing has its own lidar with separate random noise and never touches the policy's decisions. Lidar height, range
and rays are the ones in sensors.py. Prints the end-of-drive metrics after the window closes.
"""
import argparse
import json
import os

import numpy as np

from quiet import run_main
from rollout import make_policy, rollout
from sensors import LIDAR_HEIGHT_M, Lidar

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)   # data/ and benchmark/ are shared with the parent folder
DRAW_EVERY_S = 0.2
TRAIL_EVERY_M = 0.05
RED, BLUE, GREEN, GREY = [0.9, 0.1, 0.1], [0.1, 0.35, 0.95], [0.1, 0.7, 0.2], [0.35, 0.35, 0.35]


class Overlay:
    """Wraps a policy and draws on top of it; the commands pass through unchanged."""

    def __init__(self, policy, show_map=False, top=False):
        self.policy, self.name, self.show_map, self.top = policy, policy.name, show_map, top

    def reset(self, world, target):
        self.policy.reset(world, target)
        self.c, self.lidar = world.client, Lidar(world, seed=99)
        self.next_draw, self.last_trail, self.drawn_path = 0.0, None, None
        self.dots, self.map_dots, self.route = -1, -1, []
        if self.top:
            self.c.resetDebugVisualizerCamera(6.5, 0.0, -89.0, [0.0, 0.0, 0.0])

    def _points(self, pts, colour, size, replace):
        kw = {"replaceItemUniqueId": replace} if replace >= 0 else {}
        return self.c.addUserDebugPoints(pts.tolist(), [colour] * len(pts), pointSize=size, **kw)

    def _draw(self, obs):
        xy, yaw = obs["base_pos"][:2], float(obs["base_bearing"])
        ranges, hit = self.lidar.scan(xy, yaw)
        if hit.any():
            a = self.lidar.angles[hit] + yaw
            pts = np.column_stack([xy[0] + ranges[hit] * np.cos(a), xy[1] + ranges[hit] * np.sin(a),
                                   np.full(hit.sum(), LIDAR_HEIGHT_M)])
            self.dots = self._points(pts, RED, 4, self.dots)
        path = getattr(self.policy, "path", None)
        if path is not None and path is not self.drawn_path:
            for item in self.route:
                self.c.removeUserDebugItem(item)
            self.route = [self.c.addUserDebugLine([a[0], a[1], 0.03], [b[0], b[1], 0.03], BLUE, lineWidth=3)
                          for a, b in zip(path[:-1], path[1:])]
            self.drawn_path = path
        if self.show_map and hasattr(self.policy, "map") and self.policy.map.grid.occ.any():
            g = self.policy.map.grid
            cells = np.argwhere(g.occ)
            pts = np.column_stack([g.origin + (cells + 0.5) * g.res, np.full(len(cells), 0.05)])
            self.map_dots = self._points(pts, GREY, 3, self.map_dots)

    def act(self, obs):
        out = self.policy.act(obs)
        xy = obs["base_pos"][:2].copy()
        if self.last_trail is None:
            self.last_trail = xy
        elif np.linalg.norm(xy - self.last_trail) > TRAIL_EVERY_M:
            self.c.addUserDebugLine([*self.last_trail, 0.02], [*xy, 0.02], GREEN, lineWidth=3)
            self.last_trail = xy
        if obs["t"] >= self.next_draw:
            self._draw(obs)
            self.next_draw = obs["t"] + DRAW_EVERY_S
        return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", required=True, help="'expert', 'planner', 'planner:<model>.npz' or a model .npz")
    ap.add_argument("--tier", type=int, default=1)
    ap.add_argument("--index", type=int, required=True)
    ap.add_argument("--max-time", type=float, default=40.0)
    ap.add_argument("--scenes", default=os.path.join(ROOT, "benchmark", "scenes.json"))
    ap.add_argument("--top", action="store_true", help="view from above")
    ap.add_argument("--show-map", action="store_true", help="also draw the robot's occupancy map (slower)")
    a = ap.parse_args()

    with open(a.scenes) as fh:
        scenes = json.load(fh)["scenes"]
    match = [s for s in scenes if s["tier"] == a.tier and s["index"] == a.index]
    if not match:
        raise SystemExit(f"no scene with tier {a.tier} and index {a.index} in {a.scenes}")
    scene = match[0]

    print(f"tier {a.tier} #{a.index}: target {scene['target']} start {scene['start_pose']}")
    m, _ = rollout(Overlay(make_policy(a.policy), a.show_map, a.top), scene, max_time=a.max_time, use_gui=True)
    print(f"\noutcome: {m['outcome']} | final {m['final_err_cm']:.1f} cm | "
          f"closest {m['min_err_cm']:.1f} cm | time {m['time_s']:.1f} s")


if __name__ == "__main__":
    run_main(main, HERE)
