#!/usr/bin/env python3
"""Measure how the IMU is mounted, with the vehicle standing still on level ground.

Averages the IMU for a few seconds. The accelerometer then points along gravity, so
its direction gives the roll and pitch of the sensor relative to the base frame, and
the gyro average is its bias. Writes a YAML file for the collision monitor's
imu_calibration_file parameter.

    ros2 run perception_tools calibrate_imu --ros-args -p output:=imu_calibration.yaml
"""

import math

import numpy as np

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.time import Time

from sensor_msgs.msg import Imu

import tf2_ros

from perception_tools.geometry import quat_to_matrix


class ImuCalibration(Node):

    def __init__(self):
        super().__init__('calibrate_imu')
        self.duration = self.declare_parameter('duration', 10.0).value
        self.base_frame = self.declare_parameter('base_frame', 'base_footprint').value
        self.output = self.declare_parameter('output', 'imu_calibration.yaml').value
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.accel = []
        self.gyro = []
        self.rotation = None
        self.start = None
        self.done = False
        self.create_subscription(Imu, '/imu', self.on_imu, 100)
        self.get_logger().info(
            f'keep the vehicle still on level ground for {self.duration:.0f} s')

    def on_imu(self, msg):
        if self.done:
            return
        if self.rotation is None:
            try:
                tf = self.tf_buffer.lookup_transform(
                    self.base_frame, msg.header.frame_id, Time())
            except tf2_ros.TransformException:
                return
            r = tf.transform.rotation
            self.rotation = np.array(quat_to_matrix((r.x, r.y, r.z, r.w)))
            self.start = self.get_clock().now()
        a, w = msg.linear_acceleration, msg.angular_velocity
        self.accel.append(self.rotation @ np.array([a.x, a.y, a.z]))
        self.gyro.append(self.rotation @ np.array([w.x, w.y, w.z]))
        if (self.get_clock().now() - self.start).nanoseconds * 1e-9 >= self.duration:
            self.finish()

    def finish(self):
        self.done = True
        accel = np.array(self.accel)
        f = accel.mean(axis=0)
        roll = math.atan2(f[1], f[2])
        pitch = math.atan2(-f[0], math.hypot(f[1], f[2]))
        bias = np.array(self.gyro).mean(axis=0)
        noise = np.linalg.norm(accel - f, axis=1).std()
        with open(self.output, 'w') as fh:
            fh.write(f'# {len(accel)} samples over {self.duration:.0f} s\n')
            fh.write(f'roll_offset: {roll:.6f}    # rad, {math.degrees(roll):.2f} deg\n')
            fh.write(f'pitch_offset: {pitch:.6f}   # rad, {math.degrees(pitch):.2f} deg\n')
            fh.write('gyro_bias: [{:.6f}, {:.6f}, {:.6f}]   # rad/s\n'.format(*bias))
        self.get_logger().info(
            'mount roll {:.2f} deg, pitch {:.2f} deg, |f| {:.3f} m/s^2, accel noise {:.3f}; '
            'written to {}'.format(math.degrees(roll), math.degrees(pitch),
                                   np.linalg.norm(f), noise, self.output))
        if noise > 0.3:
            self.get_logger().warn('accelerometer noisy; was the vehicle moving?')


def main(argv=None):
    rclpy.init(args=argv)
    node = ImuCalibration()
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
