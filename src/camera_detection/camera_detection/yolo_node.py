#!/usr/bin/env python3
"""2D camera detection node.

Runs YOLO on the camera image and publishes vision_msgs/Detection2DArray. Falls back
to the OpenCV HOG pedestrian detector if ultralytics is not installed.
"""

import numpy as np

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node

from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose

# COCO class name -> our hazard classes
COCO_TO_CLASS = {
    'person': 'Pedestrian',
    'car': 'Vehicle',
    'truck': 'Vehicle',
    'bus': 'Vehicle',
}


class YoloNode(Node):

    def __init__(self):
        super().__init__('camera_detector')
        self.weights = self.declare_parameter('weights', 'yolov8n.pt').value
        self.conf = float(self.declare_parameter('conf', 0.4).value)
        self.bridge = CvBridge()
        self.backend, self.model = self._load()

        self.pub = self.create_publisher(Detection2DArray, '/detections/camera', 10)
        self.create_subscription(Image, '/camera/image_raw', self.on_image, 5)
        self.get_logger().info(f'camera detector ready (backend: {self.backend})')

    def _load(self):
        try:
            from ultralytics import YOLO
            model = YOLO(self.weights)
            return 'yolo', model
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(
                f'YOLO unavailable ({exc}); using HOG pedestrian fallback')
            import cv2
            hog = cv2.HOGDescriptor()
            hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            return 'hog', hog

    def on_image(self, msg):
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        if self.backend == 'yolo':
            dets = self._detect_yolo(img)
        else:
            dets = self._detect_hog(img)

        out = Detection2DArray()
        out.header = msg.header
        for cls, score, cx, cy, w, h in dets:
            det = Detection2D()
            det.header = msg.header
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = cls
            hyp.hypothesis.score = float(score)
            det.results.append(hyp)
            det.bbox.center.position.x = float(cx)
            det.bbox.center.position.y = float(cy)
            det.bbox.size_x = float(w)
            det.bbox.size_y = float(h)
            out.detections.append(det)
        self.pub.publish(out)

    def _detect_yolo(self, img):
        results = self.model.predict(img, conf=self.conf, verbose=False)
        dets = []
        for r in results:
            names = r.names
            for box in r.boxes:
                name = names[int(box.cls[0])]
                cls = COCO_TO_CLASS.get(name)
                if cls is None:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                dets.append((cls, float(box.conf[0]),
                             (x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1))
        return dets

    def _detect_hog(self, img):
        import cv2
        rects, weights = self.model.detectMultiScale(
            img, winStride=(8, 8), padding=(8, 8), scale=1.05)
        dets = []
        for (x, y, w, h), score in zip(rects, np.atleast_1d(weights)):
            if score < self.conf:
                continue
            dets.append(('Pedestrian', float(score),
                         x + w / 2.0, y + h / 2.0, float(w), float(h)))
        return dets


def main(argv=None):
    rclpy.init(args=argv)
    node = YoloNode()
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
