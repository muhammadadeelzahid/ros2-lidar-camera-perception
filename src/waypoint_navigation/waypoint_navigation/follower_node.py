#!/usr/bin/env python3
"""Waypoint follower: the FollowWaypoints action, driven by pure pursuit on odometry.

Commands go to cmd_vel_in, so they pass through the collision monitor like teleop
does. When the monitor stops the robot, the follower waits and reports why; if no
progress is made for blocked_timeout seconds the goal is aborted. It does not plan
around obstacles. Positions come from odometry, so errors grow with distance driven.
"""

import math
import threading
import time

import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node

from diagnostic_msgs.msg import DiagnosticArray
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

from perception_msgs.action import FollowWaypoints

from waypoint_navigation import pure_pursuit as pp

GATE = 'collision_monitor: gate'


def yaw_from_quat(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class Follower(Node):

    def __init__(self):
        super().__init__('waypoint_follower')
        self.speed = self.declare_parameter('speed', 0.5).value
        self.max_yaw_rate = self.declare_parameter('max_yaw_rate', 1.0).value
        self.lookahead = self.declare_parameter('lookahead', 0.6).value
        self.goal_tolerance = self.declare_parameter('goal_tolerance', 0.15).value
        self.heading_tolerance = math.radians(
            self.declare_parameter('heading_tolerance_deg', 1.0).value)
        self.decel = self.declare_parameter('decel', 0.5).value
        self.blocked_timeout = self.declare_parameter('blocked_timeout', 60.0).value
        self.rate = self.declare_parameter('rate', 20.0).value

        group = ReentrantCallbackGroup()
        self.pose = None            # (x, y, yaw) in odom
        self.velocity = (0.0, 0.0)  # (v, w) measured
        self.gate = ''
        self.lock = threading.Lock()
        self.active = None
        self.create_subscription(Odometry, '/odom', self.on_odom, 10, callback_group=group)
        self.create_subscription(DiagnosticArray, '/diagnostics', self.on_diag, 20,
                                 callback_group=group)
        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel_in', 10)
        self.server = ActionServer(
            self, FollowWaypoints, 'follow_waypoints', self.execute,
            goal_callback=lambda _: GoalResponse.ACCEPT,
            cancel_callback=lambda _: CancelResponse.ACCEPT,
            handle_accepted_callback=self.on_accepted, callback_group=group)
        self.get_logger().info('waypoint follower ready')

    def on_odom(self, msg):
        p = msg.pose.pose
        self.pose = (p.position.x, p.position.y, yaw_from_quat(p.orientation))
        self.velocity = (msg.twist.twist.linear.x, msg.twist.twist.angular.z)

    def on_diag(self, msg):
        for st in msg.status:
            if st.name == GATE:
                self.gate = st.message

    def on_accepted(self, goal_handle):
        # A new goal replaces the one being driven.
        with self.lock:
            self.active = goal_handle
        goal_handle.execute()

    def send(self, v, w):
        cmd = Twist()
        cmd.linear.x = float(v)
        cmd.angular.z = float(w)
        self.cmd_pub.publish(cmd)

    def finish(self, goal_handle, ok, message, travelled):
        self.send(0.0, 0.0)
        result = FollowWaypoints.Result(success=ok, message=message,
                                        distance_travelled=float(travelled))
        if goal_handle.is_cancel_requested:
            goal_handle.canceled()
        elif ok:
            goal_handle.succeed()
        else:
            goal_handle.abort()
        self.get_logger().info(f'goal finished: {message}')
        return result

    def execute(self, goal_handle):
        goal = goal_handle.request
        end = time.time() + 5.0
        while self.pose is None and time.time() < end:
            time.sleep(0.05)
        if self.pose is None:
            return self.finish(goal_handle, False, 'no odometry', 0.0)

        x0, y0, yaw0 = self.pose
        path = [(x0, y0)]
        for p in goal.waypoints:
            if goal.relative:
                c, s = math.cos(yaw0), math.sin(yaw0)
                path.append((x0 + c * p.x - s * p.y, y0 + s * p.x + c * p.y))
            else:
                path.append((p.x, p.y))
        heading = None
        if not math.isnan(goal.final_heading):
            heading = goal.final_heading + (yaw0 if goal.relative else 0.0)
        speed = goal.speed if goal.speed > 0.0 else self.speed
        self.get_logger().info(
            f'following {len(path) - 1} waypoints at {speed:.2f} m/s'
            + (f', then turning to {math.degrees(pp.wrap(heading)):.0f} deg' if heading is not None else ''))

        seg = 0
        driving = len(path) > 1
        travelled, last_xy = 0.0, (x0, y0)
        best, progress_time = math.inf, time.time()
        feedback = FollowWaypoints.Feedback()
        next_feedback = 0.0
        period = 1.0 / self.rate
        while rclpy.ok():
            if self.active is not goal_handle:
                return self.finish(goal_handle, False, 'replaced by a new goal', travelled)
            if goal_handle.is_cancel_requested:
                return self.finish(goal_handle, False, 'cancelled', travelled)
            x, y, yaw = self.pose
            travelled += math.dist((x, y), last_xy)
            last_xy = (x, y)

            if driving:
                seg = pp.advance(path, seg, (x, y))
                remaining = pp.remaining_distance(path, seg, (x, y))
                to_end = math.dist((x, y), path[-1])
                # Only on the last segment, so a route that ends where it began still runs.
                last = seg == len(path) - 2
                passed = last and pp.project(path, seg, (x, y)) >= 1.0
                if last and (to_end < self.goal_tolerance
                             or (passed and to_end < 2 * self.goal_tolerance)):
                    driving = False
                    remaining = 0.0
                else:
                    target = pp.lookahead_point(path, seg, (x, y), self.lookahead)
                    v_max = pp.stopping_speed(remaining, speed, self.decel)
                    v, w = pp.command((x, y, yaw), target, v_max, self.max_yaw_rate)
                    state = 'driving'
            if not driving:
                remaining = 0.0
                if heading is None:
                    return self.finish(goal_handle, True, 'reached the last waypoint', travelled)
                w, err = pp.turn_command(yaw, heading, self.max_yaw_rate)
                v = 0.0
                state = 'turning'
                if abs(err) < self.heading_tolerance:
                    return self.finish(goal_handle, True, 'reached the goal heading', travelled)
                remaining = abs(err)  # progress while turning is the shrinking error
            self.send(v, w)

            # Waiting: asked to move but not moving, usually because the monitor stopped it.
            asked = abs(v) > 0.02 or abs(w) > 0.05
            still = abs(self.velocity[0]) < 0.02 and abs(self.velocity[1]) < 0.05
            reason = ''
            if asked and still:
                state = 'waiting'
                reason = self.gate if self.gate.startswith('stopped') else 'not moving'
            if remaining < best - 0.05:
                best, progress_time = remaining, time.time()
            elif time.time() - progress_time > self.blocked_timeout:
                return self.finish(
                    goal_handle, False,
                    f'no progress for {self.blocked_timeout:.0f} s ({reason or state})', travelled)

            if time.time() >= next_feedback:
                feedback.waypoint = min(seg + 1, len(path) - 1)
                feedback.distance_remaining = float(remaining if driving else 0.0)
                feedback.state = state
                feedback.reason = reason
                goal_handle.publish_feedback(feedback)
                next_feedback = time.time() + 0.5
            time.sleep(period)
        return self.finish(goal_handle, False, 'shutting down', travelled)


def main(argv=None):
    rclpy.init(args=argv)
    node = Follower()
    executor = MultiThreadedExecutor()
    try:
        rclpy.spin(node, executor=executor)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if rclpy.ok():
            node.send(0.0, 0.0)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
