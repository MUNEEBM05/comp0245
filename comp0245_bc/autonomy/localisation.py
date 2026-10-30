"""Where has the robot got to, from wheel encoders and gyro only? A Kalman filter, in the frame the robot started in.

State: x, y, heading, gyro bias.
Predict  distance from the encoders; turning from the gyro minus its estimated bias.
Correct  when the encoders say the robot is standing still, the gyro should read zero, so what it reads is its bias.
There is no absolute position reading here, so position error can only grow with distance driven.
"""
import numpy as np

WHEEL_RADIUS = 0.11
TRACK_WIDTH = 0.66        # effective width, calibrated for skid-steer by the template (robotnik_arm_control.py)
STILL_WHEELS = 0.05       # rad/s: below this on all four wheels the robot counts as standing still


class Odometry:
    def __init__(self, x=0.0, y=0.0, yaw=0.0):
        self.s = np.array([x, y, yaw, 0.0])
        self.P = np.diag([1e-6, 1e-6, 1e-6, 0.02 ** 2])
        self.q_pos, self.q_yaw, self.q_bias = 0.02 ** 2, 0.01 ** 2, 0.001 ** 2      # growth of uncertainty, per second
        self.r_still = 0.01 ** 2                                                     # gyro noise, when standing still

    def update(self, wheels, gyro, dt):
        right, left = (wheels[0] + wheels[3]) / 2, (wheels[1] + wheels[2]) / 2
        v = WHEEL_RADIUS * (right + left) / 2
        x, y, yaw, b = self.s
        w = gyro - b
        c, s = np.cos(yaw), np.sin(yaw)
        self.s = np.array([x + v * c * dt, y + v * s * dt, yaw + w * dt, b])
        F = np.eye(4)
        F[0, 2], F[1, 2], F[2, 3] = -v * s * dt, v * c * dt, -dt
        self.P = F @ self.P @ F.T + np.diag([self.q_pos * abs(v) * dt, self.q_pos * abs(v) * dt, self.q_yaw * dt, self.q_bias * dt])
        # Correction: standing still, the gyro should read zero, so what it reads is its bias. Not done while moving: the wheels'
        # turn rate is wrong during turns because they slip (using it made heading about as bad as wheels alone, 10 degrees)
        if np.all(np.abs(wheels) < STILL_WHEELS):
            S = self.P[3, 3] + self.r_still       # the correction observes the bias, the state's fourth element
            K = self.P[:, 3] / S
            self.s = self.s + K * (gyro - self.s[3])
            self.P = self.P - np.outer(K, self.P[3, :])
        return self.s[:3].copy()


class WheelsOnly:
    """For comparison: the same position sum, with heading from the wheels instead of the gyro."""

    def __init__(self, x=0.0, y=0.0, yaw=0.0):
        self.s = np.array([x, y, yaw])

    def update(self, wheels, dt):
        right, left = (wheels[0] + wheels[3]) / 2, (wheels[1] + wheels[2]) / 2
        v, w = WHEEL_RADIUS * (right + left) / 2, WHEEL_RADIUS * (right - left) / TRACK_WIDTH
        self.s += np.array([v * np.cos(self.s[2]) * dt, v * np.sin(self.s[2]) * dt, w * dt])
        return self.s.copy()
