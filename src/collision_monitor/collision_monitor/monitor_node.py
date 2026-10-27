#!/usr/bin/env python3
"""Collision monitor and velocity gate.

Turns tracked objects into graded warnings based on distance and time to collision.
A protective field on the raw point cloud backs this up: any points in a speed-
dependent box ahead of the vehicle stop it, whether or not they were detected.
Velocity commands on cmd_vel_in are passed to cmd_vel, or replaced with zero while
either check calls for a stop.

With an IMU, the monitor also estimates how close the vehicle is to tipping over and
scales down commands whose predicted tip-over margin is too small.
"""

import math
import os

import numpy as np
import yaml

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.time import Time

from geometry_msgs.msg import Point, Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu, PointCloud2
from visualization_msgs.msg import Marker, MarkerArray

import tf2_ros

from perception_msgs.msg import (
    CollisionWarning,
    CollisionWarningArray,
    StabilityStatus,
    TrackedObjectArray,
)

from collision_monitor import field
from collision_monitor.logic import (
    LEVEL_CRITICAL,
    LEVEL_NONE,
    assess,
    circumscribed_radius,
)
from collision_monitor.attitude import LowPass
from collision_monitor.stability_monitor import StabilityMonitor

NO_TRACK = 2 ** 32 - 1  # track_id used for protective-field warnings

LEVEL_COLOR = {
    0: (0.6, 0.6, 0.6),
    1: (0.2, 0.7, 0.9),
    2: (0.95, 0.75, 0.1),
    3: (0.9, 0.1, 0.1),
}


def yaw_from_quat(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class MonitorNode(Node):

    def __init__(self):
        super().__init__('collision_monitor')
        self.ego_length = self.declare_parameter('ego_length', 0.4).value
        self.ego_width = self.declare_parameter('ego_width', 0.3).value
        self.ego_radius = circumscribed_radius(self.ego_length, self.ego_width)
        self.enable_estop = self.declare_parameter('enable_estop', True).value
        self.use_odom = self.declare_parameter('use_odom', True).value
        self.thresholds = {
            'crit_dist': self.declare_parameter('crit_dist', 0.5).value,
            'warn_dist': self.declare_parameter('warn_dist', 1.5).value,
            'info_dist': self.declare_parameter('info_dist', 3.0).value,
            'crit_ttc': self.declare_parameter('crit_ttc', 1.0).value,
            'warn_ttc': self.declare_parameter('warn_ttc', 3.0).value,
        }

        self.use_field = self.declare_parameter('use_protective_field', True).value
        self.base_frame = self.declare_parameter('base_frame', 'base_footprint').value
        self.field_min_length = self.declare_parameter('field_min_length', 0.4).value
        self.field_reaction_time = self.declare_parameter('field_reaction_time', 1.0).value
        self.field_margin = self.declare_parameter('field_margin', 0.1).value
        self.field_min_z = self.declare_parameter('field_min_z', 0.1).value
        self.field_max_z = self.declare_parameter('field_max_z', 2.0).value
        self.field_min_points = self.declare_parameter('field_min_points', 3).value
        # Stop when tracks stop arriving, e.g. the detector is still loading or died.
        self.tracks_timeout = self.declare_parameter('tracks_timeout', 0.5).value

        # Tip-over stability from the IMU. Angles are configured in degrees.
        self.use_stability = self.declare_parameter('use_stability', False).value
        self.stability_gate = self.declare_parameter('stability_gate', True).value
        self.imu_timeout = self.declare_parameter('imu_timeout', 0.2).value
        self.max_decel = self.declare_parameter('max_decel', 0.5).value
        support = self.declare_parameter(
            'support_points', [0.17, 0.0, 0.0, 0.17, -0.17, 0.0, 0.0, -0.17]).value
        cg = self.declare_parameter('cg', [0.0, 0.0, 0.12]).value
        deg = [math.radians(self.declare_parameter(name, default).value) for name, default in (
            ('stability_info_deg', 15.0), ('stability_warn_deg', 10.0),
            ('stability_crit_deg', 5.0), ('stability_min_margin_deg', 5.0))]
        self.stability = StabilityMonitor(
            np.reshape(support, (-1, 2)), cg, *deg,
            time_constant=self.declare_parameter('attitude_time_constant', 1.0).value,
            accel_gate=self.declare_parameter('attitude_accel_gate', 2.0).value,
            lowpass_hz=self.declare_parameter('accel_lowpass_hz', 5.0).value,
            response_time=self.declare_parameter('response_time', 0.5).value,
            max_accel=self.declare_parameter('max_accel', 1.0).value,
            max_decel=self.max_decel)
        calib = self.declare_parameter('imu_calibration_file', '').value
        if self.use_stability and calib:
            self._load_imu_calibration(calib)

        # Ego state (x, y, yaw, vx, vy) in the tracking frame. Without odometry the
        # tracks are already vehicle-relative, so the ego sits at the origin.
        self.ego = None if self.use_odom else (0.0, 0.0, 0.0, 0.0, 0.0)
        self.forward_speed = 0.0
        self.track_critical = False
        self.field_blocked = False
        self.field_distance = float('inf')
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.cloud_tf = {}
        self.last_reverse = None
        self.last_tracks = None
        self.last_imu = None
        self.last_imu_stamp = None
        self.last_odom = None
        self.speed_rate = LowPass(2.0)
        self.speed_accel = 0.0
        self.last_cmd = (0.0, 0.0)
        self.last_cmd_time = None
        self.command_scale = 1.0

        self.create_subscription(Odometry, '/odom', self.on_odom, 10)
        self.create_subscription(TrackedObjectArray, '/tracks', self.on_tracks, 10)
        self.create_subscription(Twist, 'cmd_vel_in', self.on_cmd_in, 10)
        if self.use_field:
            self.create_subscription(PointCloud2, '/points', self.on_cloud, 1)
        if self.use_stability:
            self.create_subscription(Imu, '/imu', self.on_imu, 50)
            self.stability_pub = self.create_publisher(StabilityStatus, '/stability', 10)
            self.create_timer(0.05, self.on_stability_timer)

        self.warn_pub = self.create_publisher(
            CollisionWarningArray, '/collision_warnings', 10)
        self.marker_pub = self.create_publisher(
            MarkerArray, '/collision_warnings/markers', 10)
        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        # Actively hold the robot stopped while an e-stop is latched.
        self.create_timer(0.05, self.on_brake_timer)
        self.get_logger().info('collision monitor ready')

    @property
    def estop(self):
        return (self.track_critical or self.field_blocked or self.tracks_stale
                or self.imu_stale)

    @property
    def tracks_stale(self):
        if self.last_tracks is None:
            return True
        age = (self.get_clock().now() - self.last_tracks).nanoseconds * 1e-9
        return age > self.tracks_timeout

    @property
    def imu_stale(self):
        if not self.use_stability:
            return False
        if self.last_imu is None:
            return True
        return (self.get_clock().now() - self.last_imu).nanoseconds * 1e-9 > self.imu_timeout

    def _load_imu_calibration(self, path):
        if not os.path.exists(path):
            self.get_logger().warn(f'IMU calibration {path} not found; assuming a level mount')
            return
        with open(path) as fh:
            c = yaml.safe_load(fh) or {}
        self.stability.calibrate(c.get('roll_offset', 0.0), c.get('pitch_offset', 0.0),
                                 c.get('gyro_bias', [0.0, 0.0, 0.0]))
        self.get_logger().info(
            'IMU mount offset roll {:.2f} deg, pitch {:.2f} deg'.format(
                math.degrees(c.get('roll_offset', 0.0)), math.degrees(c.get('pitch_offset', 0.0))))

    def on_odom(self, msg):
        stamp = Time.from_msg(msg.header.stamp).nanoseconds * 1e-9
        if self.last_odom is not None and stamp > self.last_odom[0]:
            dt = stamp - self.last_odom[0]
            rate = (msg.twist.twist.linear.x - self.last_odom[1]) / dt
            self.speed_accel = float(self.speed_rate.update([rate], dt)[0])
        self.last_odom = (stamp, msg.twist.twist.linear.x)
        self.forward_speed = msg.twist.twist.linear.x
        if not self.use_odom:
            return
        p = msg.pose.pose
        yaw = yaw_from_quat(p.orientation)
        vx, vy = msg.twist.twist.linear.x, msg.twist.twist.linear.y
        vx_odom = vx * math.cos(yaw) - vy * math.sin(yaw)
        vy_odom = vx * math.sin(yaw) + vy * math.cos(yaw)
        self.ego = (p.position.x, p.position.y, yaw, vx_odom, vy_odom)

    def on_imu(self, msg):
        tf = self._sensor_to_base(msg.header.frame_id)
        if tf is None:
            return
        R, _ = tf
        stamp = Time.from_msg(msg.header.stamp).nanoseconds * 1e-9
        dt = 0.0 if self.last_imu_stamp is None else stamp - self.last_imu_stamp
        self.last_imu_stamp = stamp
        if not 0.0 <= dt < 0.5:
            dt = 0.0
        w, a = msg.angular_velocity, msg.linear_acceleration
        self.stability.update_imu(R @ np.array([w.x, w.y, w.z]), R @ np.array([a.x, a.y, a.z]),
                                  dt, self.forward_speed, self.speed_accel)
        self.last_imu = self.get_clock().now()

    def _stability_limit(self, msg):
        """The command after the stability limit, and the scale that was applied."""
        v, w = msg.linear.x, msg.angular.z
        self.last_cmd = (v, w)
        if not self.stability.ready:
            return msg, 1.0
        v_lim, w_lim, k = self.stability.limit(v, w, self.forward_speed)
        if k >= 1.0 or not self.stability_gate:
            return msg, k
        # Slow down at no more than max_decel, so the limit itself does not pitch
        # the vehicle forward.
        now = self.get_clock().now()
        dt = 0.1 if self.last_cmd_time is None else min(
            0.2, (now - self.last_cmd_time).nanoseconds * 1e-9)
        speed = self.forward_speed
        if abs(v_lim) < abs(speed):
            v_lim = math.copysign(max(abs(v_lim), abs(speed) - self.max_decel * dt), speed)
        out = Twist()
        out.linear.x = float(v_lim)
        out.angular.z = float(w_lim)
        return out, k

    def on_cmd_in(self, msg):
        if not (self.estop and self.enable_estop):
            if self.use_stability:
                msg, self.command_scale = self._stability_limit(msg)
                self.last_cmd_time = self.get_clock().now()
            self.cmd_pub.publish(msg)
            return
        # A field stop still allows backing away; the field then checks behind.
        if not (self.track_critical or self.tracks_stale or self.imu_stale) \
                and msg.linear.x < 0.0:
            self.last_reverse = self.get_clock().now()
            self.cmd_pub.publish(msg)
        else:
            self.cmd_pub.publish(Twist())

    def on_brake_timer(self):
        if not (self.estop and self.enable_estop):
            return
        if self.tracks_stale or self.imu_stale:
            self.get_logger().warn(
                'no recent {}, holding stop'.format('tracks' if self.tracks_stale else 'IMU data'),
                throttle_duration_sec=2.0)
        elif self.last_reverse is not None and not self.track_critical:
            if (self.get_clock().now() - self.last_reverse).nanoseconds < 3e8:
                return
        self.cmd_pub.publish(Twist())

    def on_stability_timer(self):
        st = self.stability
        if not st.ready:
            return
        v, w = self.last_cmd
        if self.stability_gate:
            v, w = v * self.command_scale, w * self.command_scale
        predicted = st.predicted_margin(v, w, self.forward_speed)
        msg = StabilityStatus()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.base_frame
        msg.roll = float(st.roll)
        msg.pitch = float(st.pitch)
        msg.margin = float(st.margin)
        msg.predicted_margin = float(predicted)
        msg.load_transfer = float(st.load_transfer)
        msg.edge = st.edge
        msg.level = st.level(predicted)
        msg.command_scale = float(self.command_scale)
        self.stability_pub.publish(msg)
        self.marker_pub.publish(self._stability_markers(msg.header, msg.level))
        if msg.level == LEVEL_CRITICAL:
            self.get_logger().warn(
                'tip-over margin {:.1f} deg'.format(math.degrees(min(st.margin, predicted))),
                throttle_duration_sec=1.0)

    def _stability_markers(self, header, level):
        st = self.stability
        r, g, b = LEVEL_COLOR.get(level, (1.0, 1.0, 1.0))
        poly = Marker()
        poly.header = header
        poly.ns = 'stability'
        poly.id = 0
        poly.type = Marker.LINE_STRIP
        poly.action = Marker.ADD
        poly.pose.orientation.w = 1.0
        poly.scale.x = 0.02
        poly.color.r, poly.color.g, poly.color.b, poly.color.a = r, g, b, 1.0
        pts = list(st.polygon) + [st.polygon[0]]
        poly.points = [Point(x=float(p[0]), y=float(p[1]), z=0.01) for p in pts]
        # Net force at the CG, scaled so 1 g is 0.3 m long.
        force = Marker()
        force.header = header
        force.ns = 'stability'
        force.id = 1
        force.type = Marker.ARROW
        force.action = Marker.ADD
        force.pose.orientation.w = 1.0
        force.scale.x, force.scale.y, force.scale.z = 0.02, 0.04, 0.06
        force.color.r, force.color.g, force.color.b, force.color.a = r, g, b, 1.0
        tip = st.cg - st.force * (0.3 / 9.81)
        force.points = [Point(x=float(st.cg[0]), y=float(st.cg[1]), z=float(st.cg[2])),
                        Point(x=float(tip[0]), y=float(tip[1]), z=float(tip[2]))]
        out = MarkerArray()
        out.markers = [poly, force]
        return out

    def _sensor_to_base(self, frame):
        # The sensor mount is fixed, so the transform is looked up once and cached.
        if frame not in self.cloud_tf:
            try:
                tf = self.tf_buffer.lookup_transform(self.base_frame, frame, Time())
            except tf2_ros.TransformException:
                return None
            t, r = tf.transform.translation, tf.transform.rotation
            self.cloud_tf[frame] = (field.rotation_matrix((r.x, r.y, r.z, r.w)),
                                    np.array([t.x, t.y, t.z]))
        return self.cloud_tf[frame]

    def on_cloud(self, msg):
        tf = self._sensor_to_base(msg.header.frame_id)
        if tf is None:
            return
        R, t = tf
        pts = field.cloud_to_xyz(msg) @ R.T + t
        length = field.field_length(self.forward_speed, self.field_min_length,
                                    self.field_reaction_time)
        count, nearest = field.check_field(
            pts, self.ego_length, self.ego_width, length, self.field_margin,
            self.field_min_z, self.field_max_z, forward=self.forward_speed >= -0.01)
        self.field_blocked = count >= self.field_min_points
        self.field_distance = nearest
        if self.field_blocked and self.enable_estop:
            self.get_logger().warn(
                f'protective field blocked, obstacle {nearest:.2f} m ahead',
                throttle_duration_sec=1.0)

    def on_tracks(self, msg):
        self.last_tracks = self.get_clock().now()
        if self.ego is None:
            return
        ex, ey, eyaw, evx, evy = self.ego

        out = CollisionWarningArray()
        out.header = msg.header
        markers = MarkerArray()
        max_level = LEVEL_NONE

        for obj in msg.objects:
            rel_pos = (obj.pose.position.x - ex, obj.pose.position.y - ey)
            rel_vel = (obj.velocity.x - evx, obj.velocity.y - evy)
            obj_radius = circumscribed_radius(obj.size.x, obj.size.y)
            res = assess(rel_pos, rel_vel, self.ego_radius, obj_radius,
                         self.thresholds)
            if res['level'] == LEVEL_NONE:
                continue

            w = CollisionWarning()
            w.track_id = obj.track_id
            w.class_name = obj.class_name
            w.level = res['level']
            w.distance = float(res['distance'])
            ttc = res['ttc']
            w.time_to_collision = float(ttc) if math.isfinite(ttc) else 1e3
            # Closest point expressed in the ego frame.
            cx = math.cos(-eyaw) * rel_pos[0] - math.sin(-eyaw) * rel_pos[1]
            cy = math.sin(-eyaw) * rel_pos[0] + math.cos(-eyaw) * rel_pos[1]
            w.closest_point = Point(x=float(cx), y=float(cy), z=0.0)
            out.warnings.append(w)
            markers.markers.append(self._marker(obj, w, msg.header))
            max_level = max(max_level, res['level'])

        self.track_critical = (max_level == LEVEL_CRITICAL)
        if self.field_blocked:
            w = CollisionWarning()
            w.track_id = NO_TRACK
            w.class_name = 'Obstacle'
            w.level = LEVEL_CRITICAL
            w.distance = float(self.field_distance)
            speed = abs(self.forward_speed)
            w.time_to_collision = float(self.field_distance / speed) if speed > 1e-3 else 1e3
            w.closest_point = Point(x=float(self.ego_length / 2.0 + self.field_distance))
            out.warnings.append(w)
            max_level = LEVEL_CRITICAL
        out.max_level = max_level
        self.warn_pub.publish(out)
        self.marker_pub.publish(markers)
        if self.track_critical and self.enable_estop:
            self.get_logger().warn('EMERGENCY STOP: critical object in path',
                                   throttle_duration_sec=1.0)

    def _marker(self, obj, warning, header):
        m = Marker()
        m.header = header
        m.ns = 'warnings'
        m.id = int(obj.track_id)
        m.type = Marker.TEXT_VIEW_FACING
        m.action = Marker.ADD
        m.pose.position.x = obj.pose.position.x
        m.pose.position.y = obj.pose.position.y
        m.pose.position.z = obj.pose.position.z + 1.2
        m.pose.orientation.w = 1.0
        m.scale.z = 0.4
        r, g, b = LEVEL_COLOR.get(warning.level, (1.0, 1.0, 1.0))
        m.color.r, m.color.g, m.color.b, m.color.a = r, g, b, 1.0
        m.text = '{} {:.1f}m ttc {:.1f}s'.format(
            obj.class_name, warning.distance, warning.time_to_collision)
        m.lifetime.sec = 1
        return m


def main(argv=None):
    rclpy.init(args=argv)
    node = MonitorNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
