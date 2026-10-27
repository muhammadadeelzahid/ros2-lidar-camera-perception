"""Tip-over state of the vehicle from IMU samples, and the command limit it implies.

IMU vectors are given in the base frame. The calibration offsets describe how the
sensor actually sits relative to that frame (measured by calibrate_imu) and are
removed first.
"""

import math

import numpy as np

from collision_monitor.attitude import ComplementaryFilter, LowPass, rp_matrix
from collision_monitor.stability import (
    limit_command,
    load_transfer_ratio,
    margin_level,
    order_polygon,
    predicted_specific_force,
    stability_margin,
)


class StabilityMonitor:

    def __init__(self, support_points, cg, info, warn, crit, min_margin,
                 time_constant=1.0, accel_gate=2.0, lowpass_hz=5.0, response_time=0.5,
                 max_accel=1.0, max_decel=0.5):
        # Angles in radians. min_margin: smallest predicted margin a command may have.
        # A speed change is expected to take response_time, at no more than max_accel
        # (what the drive can do) or max_decel (what the gate allows when slowing).
        self.polygon = order_polygon(support_points)
        self.cg = np.asarray(cg, dtype=float)
        self.half_track = float(np.max(np.abs(self.polygon[:, 1])))
        self.thresholds = (info, warn, crit)
        self.min_margin = min_margin
        self.response_time = response_time
        self.max_accel = max_accel
        self.max_decel = max_decel
        self.filter = ComplementaryFilter(time_constant, accel_gate)
        self.lowpass = LowPass(lowpass_hz)
        self.mount = np.eye(3)
        self.gyro_bias = np.zeros(3)

        self.roll = 0.0
        self.pitch = 0.0
        self.force = None          # filtered specific force, base frame
        self.margin = math.inf
        self.edge = 0
        self.load_transfer = 0.0

    def calibrate(self, roll_offset, pitch_offset, gyro_bias=(0.0, 0.0, 0.0)):
        self.mount = rp_matrix(roll_offset, pitch_offset)
        self.gyro_bias = np.asarray(gyro_bias, dtype=float)

    @property
    def ready(self):
        return self.force is not None

    def update_imu(self, gyro, accel, dt, speed=0.0, speed_rate=0.0):
        gyro = self.mount @ (np.asarray(gyro, dtype=float) - self.gyro_bias)
        accel = self.mount @ np.asarray(accel, dtype=float)
        known = (speed_rate, speed * gyro[2], 0.0)
        self.roll, self.pitch = self.filter.update(gyro, accel, dt, known_accel=known)
        self.force = self.lowpass.update(accel, dt)
        self.margin, self.edge = stability_margin(self.polygon, self.cg, -self.force)
        self.load_transfer = load_transfer_ratio(
            self.force[1], self.force[2], self.cg[2], self.half_track)

    def predicted_margin(self, v, omega, speed=0.0):
        accel_x = (v - speed) / self.response_time
        if abs(v) < abs(speed):
            accel_x = math.copysign(min(abs(accel_x), self.max_decel), accel_x)
        else:
            accel_x = math.copysign(min(abs(accel_x), self.max_accel), accel_x)
        f = predicted_specific_force(self.roll, self.pitch, v, omega, accel_x)
        return stability_margin(self.polygon, self.cg, -f)[0]

    def limit(self, v, omega, speed=0.0):
        """Largest scaled-down (v, omega, k) whose predicted margin is acceptable."""
        return limit_command(v, omega, lambda a, b: self.predicted_margin(a, b, speed),
                             self.min_margin)

    def level(self, predicted=math.inf):
        return margin_level(min(self.margin, predicted), *self.thresholds)
