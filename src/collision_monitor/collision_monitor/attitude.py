"""Roll and pitch from an IMU with a complementary filter.

The gyro is integrated for fast changes and the accelerometer pulls the estimate
towards the gravity direction to remove drift. While the vehicle accelerates the
accelerometer no longer points along gravity. Acceleration known from odometry
(change of speed, and speed times yaw rate in a turn) is subtracted first; what is
left is caught only partly by reducing the accelerometer's weight in proportion to
how far the measured magnitude is from g.
"""

import math

import numpy as np

from collision_monitor.stability import GRAVITY


def accel_angles(f):
    """Roll and pitch from specific force, valid when not accelerating."""
    fx, fy, fz = f
    return math.atan2(fy, fz), math.atan2(-fx, math.hypot(fy, fz))


def euler_rates(roll, pitch, p, q, r):
    """Roll and pitch rates from body angular rates p, q, r."""
    cr, sr = math.cos(roll), math.sin(roll)
    tp = math.tan(pitch)
    return p + (q * sr + r * cr) * tp, q * cr - r * sr


def rp_matrix(roll, pitch):
    """Rotation from a frame at (roll, pitch) to the level frame, yaw ignored."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    return ry @ rx


class ComplementaryFilter:

    def __init__(self, time_constant=1.0, accel_gate=2.0):
        # time_constant (s): below it the gyro is trusted, above it the accelerometer.
        # accel_gate (m/s^2): deviation of |f| from g at which the accelerometer
        # is ignored completely.
        self.tau = time_constant
        self.gate = accel_gate
        self.roll = None
        self.pitch = None

    def update(self, gyro, accel, dt, known_accel=None):
        """known_accel: body-frame acceleration from odometry, subtracted from accel."""
        if known_accel is not None:
            accel = np.asarray(accel, dtype=float) - np.asarray(known_accel, dtype=float)
        acc_roll, acc_pitch = accel_angles(accel)
        if self.roll is None:
            self.roll, self.pitch = acc_roll, acc_pitch
            return self.roll, self.pitch
        droll, dpitch = euler_rates(self.roll, self.pitch, *gyro)
        roll = self.roll + droll * dt
        pitch = self.pitch + dpitch * dt
        weight = max(0.0, 1.0 - abs(np.linalg.norm(accel) - GRAVITY) / self.gate)
        k = dt / (self.tau + dt) * weight
        self.roll = roll + k * _wrap(acc_roll - roll)
        self.pitch = pitch + k * (acc_pitch - pitch)
        return self.roll, self.pitch


class LowPass:
    """First-order low-pass filter on a vector."""

    def __init__(self, cutoff_hz):
        self.rc = 1.0 / (2.0 * math.pi * cutoff_hz) if cutoff_hz > 0 else 0.0
        self.value = None

    def update(self, x, dt):
        x = np.asarray(x, dtype=float)
        if self.value is None or self.rc == 0.0:
            self.value = x
        else:
            a = dt / (self.rc + dt)
            self.value = self.value + a * (x - self.value)
        return self.value


def _wrap(a):
    return math.atan2(math.sin(a), math.cos(a))
