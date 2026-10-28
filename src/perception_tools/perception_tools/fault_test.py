#!/usr/bin/env python3
"""Inject sensor faults in simulation and check how the robot responds.

Sensor topics are relayed to the perception nodes through /test/*, so each one can be
stopped or degraded on cue. For every test the robot drives, the fault is switched on,
and the robot's true speed from Gazebo is measured; then the fault is cleared and the
robot should recover. The last test kills the collision monitor itself, which should
leave the drive watchdog stopping the wheels.

Run with the simulation started from scripts/fault_test.sh.
"""

import math
import os
import signal
import time

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from diagnostic_msgs.msg import DiagnosticArray
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, Imu, PointCloud2

# name, fault, commanded speed (m/s), expected response
TESTS = [
    ('LiDAR stops', 'lidar_stop', 0.3, 'stop'),
    ('LiDAR 135 deg blind', 'lidar_sector', 0.5, 'slow'),
    ('LiDAR 250 deg blind', 'lidar_blocked', 0.3, 'stop'),
    ('camera stops', 'camera_stop', 0.3, 'drive'),
    ('odometry stops', 'odom_stop', 0.3, 'stop'),
    ('IMU stops', 'imu_stop', 0.3, 'stop'),
    ('collision monitor killed', 'kill_monitor', 0.3, 'stop'),
]

DRIVE_S = 4.0
FAULT_S = 5.0
RECOVER_S = 4.0
STOPPED = 0.05      # m/s
SLOW = 0.33         # m/s, degraded_max_speed plus a little


def drop_azimuth(msg, start_deg, end_deg):
    """Copy of a PointCloud2 without the points whose azimuth is in [start, end)."""
    names = {f.name: f.offset for f in msg.fields}
    n = msg.width * msg.height
    raw = np.frombuffer(msg.data, dtype=np.uint8).reshape(n, msg.point_step)
    x = raw[:, names['x']:names['x'] + 4].copy().view('<f4').ravel()
    y = raw[:, names['y']:names['y'] + 4].copy().view('<f4').ravel()
    az = np.degrees(np.mod(np.arctan2(y, x), 2.0 * math.pi))
    keep = ~((az >= start_deg) & (az < end_deg))
    out = PointCloud2()
    out.header, out.fields, out.is_bigendian = msg.header, msg.fields, msg.is_bigendian
    out.point_step, out.is_dense = msg.point_step, msg.is_dense
    out.height, out.width = 1, int(keep.sum())
    out.row_step = out.width * out.point_step
    out.data = raw[keep].tobytes()
    return out


class FaultTest(Node):

    def __init__(self):
        super().__init__('fault_test')
        self.robot = self.declare_parameter('robot', 'ego_bot').value
        only = self.declare_parameter('tests', ['']).value
        self.tests = [t for t in TESTS if t[1] in only] if any(only) else TESTS
        self.fault = None
        self.speed = 0.0
        self.diag = {}
        self.cmd = Twist()
        self.relays = {}
        for name, msg_type, src, dst in (
                ('lidar', PointCloud2, '/points', '/test/points'),
                ('camera', Image, '/camera/image_raw', '/test/image'),
                ('odom', Odometry, '/odom', '/test/odom'),
                ('imu', Imu, '/imu', '/test/imu')):
            # Reliable on the way out, which both kinds of subscriber accept.
            pub = self.create_publisher(msg_type, dst, 10)
            self.create_subscription(
                msg_type, src, lambda m, n=name, p=pub: self.relay(n, p, m),
                qos_profile_sensor_data if name in ('lidar', 'camera') else 50)
        self.create_subscription(ModelStates, '/gazebo/model_states', self.on_states, 10)
        self.create_subscription(DiagnosticArray, '/diagnostics', self.on_diag, 50)
        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel_in', 10)
        self.create_timer(0.1, lambda: self.cmd_pub.publish(self.cmd))

    def relay(self, name, pub, msg):
        f = self.fault or ''
        if f == f'{name}_stop':
            return
        if name == 'lidar' and f == 'lidar_sector':
            msg = drop_azimuth(msg, 45.0, 180.0)
        elif name == 'lidar' and f == 'lidar_blocked':
            msg = drop_azimuth(msg, 0.0, 250.0)
        pub.publish(msg)

    def on_states(self, msg):
        if self.robot in msg.name:
            t = msg.twist[msg.name.index(self.robot)].linear
            self.speed = math.hypot(t.x, t.y)

    def on_diag(self, msg):
        for st in msg.status:
            self.diag[st.name] = st.message

    def spin_for(self, seconds, record=None):
        end = time.time() + seconds
        while time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.02)
            if record is not None:
                record.append((time.time(), self.speed))

    def kill_monitor(self):
        """SIGKILL the collision monitor process, found through /proc."""
        found = False
        for pid in filter(str.isdigit, os.listdir('/proc')):
            try:
                with open(f'/proc/{pid}/cmdline', 'rb') as fh:
                    args = fh.read().split(b'\0')
            except OSError:
                continue
            if int(pid) != os.getpid() and any(
                    a.endswith(b'collision_monitor/monitor_node') for a in args):
                os.kill(int(pid), signal.SIGKILL)
                found = True
        return found

    def run_test(self, name, fault, speed, expect, direction):
        self.cmd = Twist()
        self.cmd.linear.x = direction * speed
        self.spin_for(DRIVE_S)
        moving = self.speed
        t0 = time.time()
        if fault == 'kill_monitor':
            if not self.kill_monitor():
                self.get_logger().error('collision monitor process not found')
        self.fault = fault
        samples = []
        self.spin_for(FAULT_S, samples)
        gate = self.diag.get('collision_monitor: gate', '')
        camera = self.diag.get('fusion_tracker: camera', '')
        late = [s for t, s in samples if t - t0 > FAULT_S - 2.0]
        stop_t = next((t - t0 for t, s in samples if s < STOPPED), None)
        self.get_logger().info('speed after the fault: ' + ' '.join(
            f'{t - t0:.1f}s:{s:.2f}' for t, s in samples[::25]))
        worst = max(late) if late else math.nan
        if expect == 'stop':
            ok = worst < STOPPED
        elif expect == 'slow':
            ok = STOPPED < np.mean(late) < SLOW
        else:
            ok = np.mean(late) > 0.8 * speed
        self.fault = None if fault != 'kill_monitor' else fault
        self.cmd = Twist()
        self.spin_for(RECOVER_S)
        return {
            'test': name, 'expected': expect, 'pass': bool(ok),
            'speed before (m/s)': round(moving, 2),
            'speed during, last 2 s (m/s)': round(float(np.mean(late)), 2),
            'time to stop (s)': round(stop_t, 2) if stop_t is not None and expect == 'stop' else '-',
            'monitor': gate, 'camera': camera if fault == 'camera_stop' else '',
        }

    def run_all(self):
        # Start once the whole pipeline is up and the monitor passes commands.
        end = time.time() + 120.0
        while self.diag.get('collision_monitor: gate') != 'passing commands':
            if time.time() > end:
                raise RuntimeError('collision monitor never started passing commands: '
                                   + str(self.diag.get('collision_monitor: gate')))
            self.spin_for(1.0)
        rows = []
        for i, (name, fault, speed, expect) in enumerate(self.tests):
            row = self.run_test(name, fault, speed, expect, 1 if i % 2 == 0 else -1)
            self.get_logger().info(str(row))
            rows.append(row)
        print()
        for r in rows:
            print('{:26s} {:6s} {:5s} before {:.2f} during {:.2f} stop {}  {} {}'.format(
                r['test'], r['expected'], 'PASS' if r['pass'] else 'FAIL',
                r['speed before (m/s)'], r['speed during, last 2 s (m/s)'],
                r['time to stop (s)'], r['monitor'], r['camera']))
        print(f"{sum(r['pass'] for r in rows)}/{len(rows)} passed")


def main(argv=None):
    rclpy.init(args=argv)
    node = FaultTest()
    try:
        node.run_all()
    finally:
        node.cmd_pub.publish(Twist())
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
