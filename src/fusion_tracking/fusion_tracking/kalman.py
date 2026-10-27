"""Constant-velocity Kalman filter for a 3D object center.

State:       [x, y, z, vx, vy, vz]
Measurement: [x, y, z]
"""

import numpy as np


class ConstantVelocityKF:

    def __init__(self, xyz, pos_var=0.5, vel_var=10.0,
                 process_var=1.0, meas_var=0.2):
        self.x = np.array([xyz[0], xyz[1], xyz[2], 0.0, 0.0, 0.0], dtype=float)
        self.P = np.diag([pos_var, pos_var, pos_var,
                          vel_var, vel_var, vel_var]).astype(float)
        self.q = float(process_var)
        self.r = float(meas_var)
        self.H = np.zeros((3, 6))
        self.H[0, 0] = self.H[1, 1] = self.H[2, 2] = 1.0

    def predict(self, dt):
        if dt <= 0.0:
            return
        F = np.eye(6)
        F[0, 3] = F[1, 4] = F[2, 5] = dt
        # White-acceleration process noise.
        q = self.q
        dt2, dt3, dt4 = dt * dt, dt ** 3, dt ** 4
        qp = q * dt4 / 4.0
        qpv = q * dt3 / 2.0
        qv = q * dt2
        Q = np.zeros((6, 6))
        for i in range(3):
            Q[i, i] = qp
            Q[i, i + 3] = qpv
            Q[i + 3, i] = qpv
            Q[i + 3, i + 3] = qv
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update(self, xyz):
        z = np.asarray(xyz, dtype=float)
        R = np.eye(3) * self.r
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P

    @property
    def position(self):
        return self.x[:3].copy()

    @property
    def velocity(self):
        return self.x[3:].copy()
