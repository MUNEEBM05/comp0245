"""Simulated sensors: the simulator's true state plus the errors a real sensor would have.

Wheel encoders: measured wheel speeds + noise. They cannot see the base sliding, because the wheels do not turn when it slides.
IMU, on the base: gyro = true yaw rate + noise + a slowly wandering bias.
Lidar: one horizontal scan from a mast at 1.0 m on the base, above the stowed arm (highest point 0.81 m), so the
robot does not see itself. Objects lower than the scan are invisible to it: a stated limitation of a one-plane lidar.
"""
import numpy as np

LIDAR_HEIGHT_M = 1.0
LIDAR_RANGE_M = 6.0
LIDAR_RAYS = 360
LIDAR_NOISE_M = 0.01
ENCODER_NOISE = 0.02        # rad/s, on each wheel speed
GYRO_NOISE = 0.01           # rad/s
GYRO_BIAS_START = 0.01      # rad/s, standard deviation of the bias at the start
GYRO_BIAS_WALK = 0.001      # rad/s per square root of a second


class Lidar:
    def __init__(self, world, seed=0, rays=LIDAR_RAYS, max_range=LIDAR_RANGE_M, noise=LIDAR_NOISE_M):
        self.client, self.max_range, self.noise = world.client, max_range, noise
        self.angles = np.linspace(-np.pi, np.pi, rays, endpoint=False)    # in the robot's frame, 0 = straight ahead
        self.rng = np.random.default_rng([seed, 7])

    def scan(self, base_xy, yaw):
        """Returns (ranges in m, hit mask). A ray that hits nothing reports max_range."""
        a = self.angles + yaw
        z = np.full_like(a, LIDAR_HEIGHT_M)
        origin = np.column_stack([np.full_like(a, base_xy[0]), np.full_like(a, base_xy[1]), z])
        end = origin + self.max_range * np.column_stack([np.cos(a), np.sin(a), np.zeros_like(a)])
        res = self.client.rayTestBatch(origin.tolist(), end.tolist())
        frac = np.array([r[2] for r in res])
        hit = np.array([r[0] >= 0 for r in res])
        ranges = np.where(hit, frac * self.max_range + self.rng.normal(0, self.noise, len(a)), self.max_range)
        return np.clip(ranges, 0.0, self.max_range), hit


class WheelEncoders:
    def __init__(self, seed=0, noise=ENCODER_NOISE):
        self.rng, self.noise = np.random.default_rng([seed, 8]), noise

    def read(self, qd_mes):
        return np.asarray(qd_mes[:4], float) + self.rng.normal(0, self.noise, 4)    # front_right, front_left, back_left, back_right


class Gyro:
    def __init__(self, world, seed=0, noise=GYRO_NOISE, bias_start=GYRO_BIAS_START, bias_walk=GYRO_BIAS_WALK):
        self.client, self.rid, self.noise, self.bias_walk = world.client, world.rid, noise, bias_walk
        self.rng = np.random.default_rng([seed, 9])
        self.bias = float(self.rng.normal(0, bias_start))

    def read(self, dt):
        self.bias += float(self.rng.normal(0, self.bias_walk * np.sqrt(dt)))
        return float(self.client.getBaseVelocity(self.rid)[1][2]) + self.bias + float(self.rng.normal(0, self.noise))


class Accelerometer:
    """Sideways and forward acceleration of the base, in its own frame, from the change in true velocity + noise + bias. Level floor,
    so gravity does not enter these two axes."""

    def __init__(self, world, seed=0, noise=0.02, bias=0.01):
        self.client, self.rid, self.noise = world.client, world.rid, noise
        self.rng = np.random.default_rng([seed, 10])
        self.bias, self.last = self.rng.normal(0, bias, 2), None

    def read(self, dt):
        v = np.array(self.client.getBaseVelocity(self.rid)[0][:2])
        a = np.zeros(2) if self.last is None else (v - self.last) / dt
        self.last = v
        _, q = self.client.getBasePositionAndOrientation(self.rid)
        yaw = self.client.getEulerFromQuaternion(q)[2]
        c, s = np.cos(yaw), np.sin(yaw)
        return np.array([c * a[0] + s * a[1], -s * a[0] + c * a[1]]) + self.bias + self.rng.normal(0, self.noise, 2)
