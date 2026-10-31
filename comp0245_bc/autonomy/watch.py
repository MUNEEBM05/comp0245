"""Watch one scene drive in the 3D viewer, with what the robot senses and plans drawn on top.

    export LIBGL_ALWAYS_SOFTWARE=1      # needed for the viewer on WSL at the moment
    python watch.py --policy planner --scenes scenes/obstacles_val.json --tier 14 --index 0 --max-time 90
    python watch.py --policy planner:trained/vR_ik.npz --scenes scenes/obstacles_val.json --tier 12 --index 3 --max-time 90 --top
    python watch.py --policy trained/vR_ik.npz --tier 1 --index 40          # a model on a benchmark scene

Drawn:  red dots    lidar hits
        blue line   the planned route (planner policies only)
        green line  the path the robot has actually driven
        grey dots   cells on the robot's map that count as occupied (planner policies only, with --show-map)

--camera opens a second window: the depth camera's picture, a box around what it detects as the target, and its error.
--camera-frames DIR also saves those views as pictures.

The drawing has its own lidar with separate random noise and never touches the policy's decisions. Lidar height, range
and rays are the ones in sensors.py. Prints the end-of-drive metrics after the window closes.
"""
import argparse
import json
import os

import numpy as np

from quiet import run_main
from rollout import make_policy, rollout
from camera import TargetCamera
from sensors import LIDAR_HEIGHT_M, Lidar

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)   # data/ and benchmark/ are shared with the parent folder
DRAW_EVERY_S = 0.2
TRAIL_EVERY_M = 0.05
RED, BLUE, GREEN, GREY = [0.9, 0.1, 0.1], [0.1, 0.35, 0.95], [0.1, 0.7, 0.2], [0.35, 0.35, 0.35]


class CameraWindow:
    """What the camera sees, as a separate window. Uses its own camera object, so it cannot change the drive."""

    def __init__(self, world, target, frames=None):
        import matplotlib.pyplot as plt
        self.plt, self.target, self.frames, self.n = plt, np.asarray(target, float), frames, 0
        self.cam = TargetCamera(world, seed=98)
        if frames:
            os.makedirs(frames, exist_ok=True)
        else:
            plt.switch_backend("TkAgg")
            plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(6.4, 5.2))
        self.fig.canvas.manager.set_window_title("Robot camera") if not frames else None
        self.im = self.ax.imshow(np.zeros((self.cam.h, self.cam.w, 3), np.uint8))
        self.box = plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor="lime", linewidth=2, visible=False)
        self.ax.add_patch(self.box)
        self.ax.set_axis_off()

    def update(self, xy, yaw, t):
        rgb, _ = self.cam.image(xy, yaw)
        self.im.set_data(rgb)
        r, g, b = rgb[..., 0].astype(int), rgb[..., 1].astype(int), rgb[..., 2].astype(int)
        v, u = np.nonzero((r > 120) & (g < 60) & (b < 60))
        m, n = self.cam.measure(xy, yaw)
        if m is not None:
            self.box.set_bounds(u.min() - 3, v.min() - 3, u.max() - u.min() + 6, v.max() - v.min() + 6)
            self.box.set_visible(True)
            err = 100 * np.linalg.norm(TargetCamera.to_world(m, xy, yaw) - self.target)
            dist = np.linalg.norm(self.target[:2] - xy)
            self.ax.set_title(f"t = {t:.1f} s | target seen, {n} pixels | {dist:.2f} m away | error {err:.1f} cm", fontsize=10)
        else:
            self.box.set_visible(False)
            self.ax.set_title(f"t = {t:.1f} s | target not seen", fontsize=10)
        if self.frames:
            self.fig.savefig(os.path.join(self.frames, f"view_{self.n:04d}.png"), dpi=80)
        else:
            self.plt.pause(0.001)
        self.n += 1


class Overlay:
    """Wraps a policy and draws on top of it; the commands pass through unchanged."""

    def __init__(self, policy, show_map=False, top=False, camera=False, frames=None):
        self.policy, self.name, self.show_map, self.top = policy, policy.name, show_map, top
        self.camera, self.frames = camera, frames

    def reset(self, world, target):
        self.policy.reset(world, target)
        self.c, self.lidar = world.client, Lidar(world, seed=99)
        self.next_draw, self.last_trail, self.drawn_path = 0.0, None, None
        self.dots, self.map_dots, self.route = -1, -1, []
        self.view = CameraWindow(world, target, self.frames) if self.camera else None
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
            if self.view:
                self.view.update(obs["base_pos"][:2], float(obs["base_bearing"]), obs["t"])
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
    ap.add_argument("--camera", action="store_true", help="show the depth camera's view in a second window")
    ap.add_argument("--camera-frames", default=None, help="save the camera's view as pictures here, instead of a window")
    a = ap.parse_args()

    with open(a.scenes) as fh:
        scenes = json.load(fh)["scenes"]
    match = [s for s in scenes if s["tier"] == a.tier and s["index"] == a.index]
    if not match:
        raise SystemExit(f"no scene with tier {a.tier} and index {a.index} in {a.scenes}")
    scene = match[0]

    print(f"tier {a.tier} #{a.index}: target {scene['target']} start {scene['start_pose']}")
    m, _ = rollout(Overlay(make_policy(a.policy), a.show_map, a.top, a.camera or bool(a.camera_frames), a.camera_frames), scene, max_time=a.max_time, use_gui=True)
    print(f"\noutcome: {m['outcome']} | final {m['final_err_cm']:.1f} cm | "
          f"closest {m['min_err_cm']:.1f} cm | time {m['time_s']:.1f} s")


if __name__ == "__main__":
    run_main(main, HERE)
