"""Simulated depth camera that finds the red target: colour threshold + depth -> target position in the robot's frame.

Mounted 30 cm to the left of the centre line, because on the centre line the stowed arm hides the target when parked.

The camera sees only pixels. The true target position is used by the simulator to draw the marker and, in check_camera.py, to score.
Depth noise grows with distance squared, as on stereo depth cameras: my chosen values, not a measured sensor.
"""
from _paths import ROOT, find_result  # noqa: F401  (project folder; lets the code folders import each other)
import numpy as np

TARGET_RADIUS_M = 0.03
DEPTH_NOISE_BASE_M, DEPTH_NOISE_QUAD = 0.002, 0.002   # per-pixel standard deviation: base + quad * depth^2
NEAR, FAR = 0.1, 8.0
MIN_FIT_PIXELS = 30        # below this the ball is too small for a surface fit: use the average, moved back by the radius


def fit_sphere(pts, centre, radius=TARGET_RADIUS_M, steps=5):
    """Centre of a ball of known size from the part of its surface that shows. Unlike an average, a partly hidden ball does not pull it
    towards the visible side. Gauss-Newton on (distance to centre - radius)."""
    c = np.asarray(centre, float).copy()
    for _ in range(steps):
        d = pts - c
        n = np.maximum(np.linalg.norm(d, axis=1), 1e-6)
        J = -d / n[:, None]
        step, *_ = np.linalg.lstsq(J, radius - n, rcond=None)
        c += np.clip(step, -0.02, 0.02)
    return c


def add_target_marker(world, target):
    """The target as a visible object. Visual only: no collision, so the lidar and the robot pass through it."""
    c = world.client
    vis = c.createVisualShape(c.GEOM_SPHERE, radius=TARGET_RADIUS_M, rgbaColor=[1.0, 0.0, 0.0, 1.0])
    return c.createMultiBody(baseMass=0, baseVisualShapeIndex=vis, basePosition=[float(v) for v in target])


class TargetCamera:
    def __init__(self, world, forward=0.1, left=0.3, height=1.0, pitch_deg=30.0, width=320, height_px=240, fov_deg=70.0, seed=0, noise=True):
        self.client, self.w, self.h, self.fov = world.client, width, height_px, np.radians(fov_deg)
        self.forward, self.left, self.mount_z, self.pitch = forward, left, height, np.radians(pitch_deg)
        self.rng, self.noise = np.random.default_rng([seed, 31]), noise
        self.proj = self.client.computeProjectionMatrixFOV(fov_deg, width / height_px, NEAR, FAR)
        self.f = (height_px / 2) / np.tan(self.fov / 2)

    def image(self, xy, yaw):
        """Render from the camera's true pose. Returns rgb and metric depth."""
        c, s = np.cos(yaw), np.sin(yaw)
        eye = np.array([xy[0] + self.forward * c - self.left * s, xy[1] + self.forward * s + self.left * c, self.mount_z])
        look = np.array([c * np.cos(self.pitch), s * np.cos(self.pitch), -np.sin(self.pitch)])
        up = np.array([c * np.sin(self.pitch), s * np.sin(self.pitch), np.cos(self.pitch)])
        view = self.client.computeViewMatrix(eye.tolist(), (eye + look).tolist(), up.tolist())
        _, _, rgba, depth, _ = self.client.getCameraImage(self.w, self.h, view, self.proj, renderer=self.client.ER_TINY_RENDERER)
        rgba = np.asarray(rgba, np.uint8).reshape(self.h, self.w, 4)
        d = np.asarray(depth, float).reshape(self.h, self.w)
        return rgba[..., :3], FAR * NEAR / (FAR - (FAR - NEAR) * d)

    def measure(self, xy, yaw):
        """Target centre in the robot's frame (forward, left, up), or None. xy, yaw = the camera's real pose, used only to render."""
        rgb, depth = self.image(xy, yaw)
        r, g, b = rgb[..., 0].astype(int), rgb[..., 1].astype(int), rgb[..., 2].astype(int)
        mask = (r > 120) & (g < 60) & (b < 60)
        if mask.sum() < 3:
            return None, int(mask.sum())
        v, u = np.nonzero(mask)
        z = depth[v, u]
        if self.noise:
            z = z + self.rng.normal(0, DEPTH_NOISE_BASE_M + DEPTH_NOISE_QUAD * z ** 2)
        # Pixel -> point in the camera's frame (x right, y down, z along the view), then through the mount into the robot's frame
        x_c, y_c = (u - self.w / 2) * z / self.f, (v + 1 - self.h / 2) * z / self.f     # pixel offsets of the renderer, measured without noise
        pts = np.column_stack([x_c, y_c, z])
        ray = np.median(pts, axis=0)
        ray = ray * (1 + TARGET_RADIUS_M * 0.75 / np.linalg.norm(ray))      # points lie on the near surface: move to the centre
        if len(pts) >= MIN_FIT_PIXELS:
            ray = fit_sphere(pts, ray)
        cp, sp = np.cos(self.pitch), np.sin(self.pitch)
        fwd = self.forward + cp * ray[2] - sp * ray[1]
        up = self.mount_z - sp * ray[2] - cp * ray[1]
        return np.array([fwd, self.left - ray[0], up]), int(mask.sum())

    @staticmethod
    def to_world(meas, xy, yaw):
        c, s = np.cos(yaw), np.sin(yaw)
        return np.array([xy[0] + c * meas[0] - s * meas[1], xy[1] + s * meas[0] + c * meas[1], meas[2]])
