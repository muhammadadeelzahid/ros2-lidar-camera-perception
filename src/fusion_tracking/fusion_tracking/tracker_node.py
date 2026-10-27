#!/usr/bin/env python3
"""Fusion + tracking node.

Consumes 3D LiDAR detections and 2D camera detections, confirms each LiDAR detection's
class by projecting it into the image, transforms detections into a stable tracking
frame, runs the multi-object tracker, and publishes tracked objects and RViz markers.
"""

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.time import Time

from sensor_msgs.msg import CameraInfo
from vision_msgs.msg import Detection2DArray, Detection3DArray
from visualization_msgs.msg import Marker, MarkerArray

import tf2_ros

from perception_msgs.msg import TrackedObject, TrackedObjectArray

from fusion_tracking.fusion import confirm_class, project_point
from fusion_tracking.tracker import Detection, MultiObjectTracker
from fusion_tracking import transforms as tfx

CLASS_COLOR = {
    'Pedestrian': (0.1, 0.9, 0.2),
    'Vehicle': (0.1, 0.5, 0.9),
}


class TrackerNode(Node):

    def __init__(self):
        super().__init__('fusion_tracker')
        self.tracking_frame = self.declare_parameter(
            'tracking_frame', 'odom').value
        self.camera_frame = self.declare_parameter(
            'camera_frame', 'camera_optical_frame').value
        gate = self.declare_parameter('gate_distance', 2.0).value
        max_age = self.declare_parameter('max_age', 1.0).value
        min_hits = self.declare_parameter('min_hits', 3).value

        self.tracker = MultiObjectTracker(
            gate_distance=gate, max_age=max_age, min_hits=min_hits)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.cam_dets = []      # (cls, score, cx, cy, w, h)
        self.K = None
        self.last_stamp = None

        self.create_subscription(Detection2DArray, '/detections/camera',
                                 self.on_camera, 5)
        self.create_subscription(CameraInfo, '/camera/camera_info',
                                 self.on_caminfo, 5)
        self.create_subscription(Detection3DArray, '/detections/lidar',
                                 self.on_lidar, 5)

        self.track_pub = self.create_publisher(
            TrackedObjectArray, '/tracks', 10)
        self.marker_pub = self.create_publisher(
            MarkerArray, '/tracks/markers', 10)
        self.get_logger().info('fusion + tracking ready')

    def on_caminfo(self, msg):
        if msg.header.frame_id:
            self.camera_frame = msg.header.frame_id
        k = msg.k
        self.K = [[k[0], k[1], k[2]], [k[3], k[4], k[5]], [k[6], k[7], k[8]]]

    def on_camera(self, msg):
        dets = []
        for d in msg.detections:
            if not d.results:
                continue
            cls = d.results[0].hypothesis.class_id
            score = d.results[0].hypothesis.score
            b = d.bbox
            dets.append((cls, score, b.center.position.x, b.center.position.y,
                         b.size_x, b.size_y))
        self.cam_dets = dets

    def _lookup(self, target, source, stamp):
        try:
            return self.tf_buffer.lookup_transform(target, source, Time())
        except tf2_ros.TransformException:
            return None

    def on_lidar(self, msg):
        stamp = Time.from_msg(msg.header.stamp)
        dt = 0.1
        if self.last_stamp is not None:
            dt = max(1e-3, (stamp - self.last_stamp).nanoseconds * 1e-9)
        self.last_stamp = stamp

        src = msg.header.frame_id
        to_track = self._lookup(self.tracking_frame, src, stamp)
        to_cam = self._lookup(self.camera_frame, src, stamp)
        if to_track is None:
            self.get_logger().warn(
                f'no transform {self.tracking_frame}<-{src}; skipping frame',
                throttle_duration_sec=2.0)
            return
        t_track, q_track = tfx.transform_from_msg(to_track)

        detections = []
        for d in msg.detections:
            if not d.results:
                continue
            cls = d.results[0].hypothesis.class_id
            score = d.results[0].hypothesis.score
            c = d.bbox.center
            p_src = np.array([c.position.x, c.position.y, c.position.z])
            yaw_src = tfx.yaw_from_quat(
                (c.orientation.x, c.orientation.y, c.orientation.z, c.orientation.w))

            # Camera-LiDAR fusion: confirm class by projecting into the image.
            if to_cam is not None and self.K is not None and self.cam_dets:
                t_cam, q_cam = tfx.transform_from_msg(to_cam)
                p_cam = tfx.transform_point(t_cam, q_cam, p_src)
                cls, score = confirm_class(
                    project_point(self.K, p_cam), cls, score, self.cam_dets)

            p_track = tfx.transform_point(t_track, q_track, p_src)
            yaw_track = tfx.transform_yaw(q_track, yaw_src)
            size = (d.bbox.size.x, d.bbox.size.y, d.bbox.size.z)
            detections.append(Detection(cls, score, p_track, size, yaw_track))

        self.tracker.step(detections, dt)
        self.publish(msg.header.stamp)

    def publish(self, stamp):
        tracks = self.tracker.confirmed()
        arr = TrackedObjectArray()
        arr.header.stamp = stamp
        arr.header.frame_id = self.tracking_frame
        markers = MarkerArray()

        for t in tracks:
            pos = t.position
            vel = t.velocity
            obj = TrackedObject()
            obj.track_id = int(t.id)
            obj.class_name = t.class_name
            obj.score = float(t.score)
            obj.pose.position.x = float(pos[0])
            obj.pose.position.y = float(pos[1])
            obj.pose.position.z = float(pos[2])
            qx, qy, qz, qw = tfx.quat_from_yaw(t.heading)
            obj.pose.orientation.x = qx
            obj.pose.orientation.y = qy
            obj.pose.orientation.z = qz
            obj.pose.orientation.w = qw
            obj.size.x = float(t.size[0])
            obj.size.y = float(t.size[1])
            obj.size.z = float(t.size[2])
            obj.velocity.x = float(vel[0])
            obj.velocity.y = float(vel[1])
            obj.velocity.z = float(vel[2])
            obj.age_seconds = float(t.age)
            arr.objects.append(obj)
            markers.markers.append(self._box_marker(t, stamp))

        self.track_pub.publish(arr)
        self.marker_pub.publish(markers)

    def _box_marker(self, t, stamp):
        m = Marker()
        m.header.frame_id = self.tracking_frame
        m.header.stamp = stamp
        m.ns = 'tracks'
        m.id = int(t.id)
        m.type = Marker.CUBE
        m.action = Marker.ADD
        pos = t.position
        m.pose.position.x = float(pos[0])
        m.pose.position.y = float(pos[1])
        m.pose.position.z = float(pos[2])
        qx, qy, qz, qw = tfx.quat_from_yaw(t.heading)
        m.pose.orientation.x = qx
        m.pose.orientation.y = qy
        m.pose.orientation.z = qz
        m.pose.orientation.w = qw
        m.scale.x = max(0.1, float(t.size[0]))
        m.scale.y = max(0.1, float(t.size[1]))
        m.scale.z = max(0.1, float(t.size[2]))
        r, g, b = CLASS_COLOR.get(t.class_name, (0.8, 0.8, 0.1))
        m.color.r, m.color.g, m.color.b, m.color.a = r, g, b, 0.5
        m.lifetime.sec = 1
        return m


def main(argv=None):
    rclpy.init(args=argv)
    node = TrackerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
