#!/usr/bin/env python3
"""Evaluate the perception pipeline against simulator ground truth.

Compares /tracks with the true object poses from /gazebo/model_states, both in the
ego frame. On Ctrl-C it prints recall, precision, position error and ID switches, and
writes a plot of position error against range.
"""

import argparse
import math
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

import numpy as np  # noqa: E402

import rclpy  # noqa: E402
from rclpy.node import Node  # noqa: E402

from gazebo_msgs.msg import ModelStates  # noqa: E402
from nav_msgs.msg import Odometry  # noqa: E402

from perception_msgs.msg import TrackedObjectArray  # noqa: E402

from perception_tools.classes import classify  # noqa: E402
from perception_tools.geometry import yaw_from_quaternion  # noqa: E402

EGO_MODEL = 'ego_bot'


def to_ego(ex, ey, eyaw, px, py):
    dx, dy = px - ex, py - ey
    c, s = math.cos(-eyaw), math.sin(-eyaw)
    return (c * dx - s * dy, s * dx + c * dy)


class Evaluator(Node):

    def __init__(self, match_gate, max_range, output_dir):
        super().__init__('eval_metrics')
        self.match_gate = match_gate
        self.max_range = max_range
        self.output_dir = output_dir

        self.ego_world = None   # (x, y, yaw) from model_states
        self.ego_odom = None    # (x, y, yaw) from /odom
        self.gt = {}            # name -> (x, y) ego frame
        self.last_match = {}    # gt name -> track_id

        self.errors = []        # (range, position_error)
        self.id_switches = 0
        self.frames = 0
        self.recall_num = 0
        self.recall_den = 0
        self.prec_num = 0
        self.prec_den = 0

        self.create_subscription(ModelStates, '/gazebo/model_states',
                                 self.on_states, 10)
        self.create_subscription(Odometry, '/odom', self.on_odom, 10)
        self.create_subscription(TrackedObjectArray, '/tracks', self.on_tracks, 10)
        self.get_logger().info('evaluating; press Ctrl-C for the report')

    def on_states(self, msg):
        if EGO_MODEL not in msg.name:
            return
        ep = msg.pose[msg.name.index(EGO_MODEL)]
        eyaw = yaw_from_quaternion(ep.orientation.x, ep.orientation.y,
                                   ep.orientation.z, ep.orientation.w)
        self.ego_world = (ep.position.x, ep.position.y, eyaw)
        gt = {}
        for name, pose in zip(msg.name, msg.pose):
            if classify(name) is None:
                continue
            rel = to_ego(ep.position.x, ep.position.y, eyaw,
                         pose.position.x, pose.position.y)
            if math.hypot(*rel) <= self.max_range:
                gt[name] = rel
        self.gt = gt

    def on_odom(self, msg):
        p = msg.pose.pose
        yaw = yaw_from_quaternion(p.orientation.x, p.orientation.y,
                                  p.orientation.z, p.orientation.w)
        self.ego_odom = (p.position.x, p.position.y, yaw)

    def on_tracks(self, msg):
        if self.ego_odom is None or not self.gt:
            return
        ex, ey, eyaw = self.ego_odom
        tracks = []
        for o in msg.objects:
            rel = to_ego(ex, ey, eyaw, o.pose.position.x, o.pose.position.y)
            tracks.append((o.track_id, rel))

        self.frames += 1
        self.recall_den += len(self.gt)
        self.prec_den += len(tracks)

        used = set()
        for name, g in self.gt.items():
            best, best_d = None, self.match_gate
            for i, (tid, rel) in enumerate(tracks):
                if i in used:
                    continue
                d = math.hypot(g[0] - rel[0], g[1] - rel[1])
                if d < best_d:
                    best, best_d = i, d
            if best is None:
                continue
            used.add(best)
            tid = tracks[best][0]
            self.recall_num += 1
            self.errors.append((math.hypot(*g), best_d))
            if name in self.last_match and self.last_match[name] != tid:
                self.id_switches += 1
            self.last_match[name] = tid
        self.prec_num += len(used)

    def report(self):
        os.makedirs(self.output_dir, exist_ok=True)
        recall = self.recall_num / self.recall_den if self.recall_den else 0.0
        precision = self.prec_num / self.prec_den if self.prec_den else 0.0
        if self.errors:
            err = np.array([e[1] for e in self.errors])
            self.get_logger().info(
                f'frames {self.frames} | recall {recall:.2f} | precision '
                f'{precision:.2f} | pos-err mean {err.mean():.3f} m max '
                f'{err.max():.3f} m | id-switches {self.id_switches}')
        else:
            self.get_logger().info('no matched detections recorded')

        if self.errors:
            arr = np.array(self.errors)
            plt.figure(figsize=(7, 5))
            plt.scatter(arr[:, 0], arr[:, 1], s=6)
            plt.xlabel('range to object [m]')
            plt.ylabel('position error [m]')
            plt.title('Distance-estimation error vs range')
            path = os.path.join(self.output_dir, 'error_vs_range.png')
            plt.savefig(path, dpi=120)
            self.get_logger().info(f'wrote {path}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--match-gate', type=float, default=2.0)
    parser.add_argument('--max-range', type=float, default=15.0)
    parser.add_argument('--output-dir', default='analysis_output')
    args, ros_args = parser.parse_known_args(argv)

    rclpy.init(args=ros_args)
    node = Evaluator(args.match_gate, args.max_range, args.output_dir)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.report()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
