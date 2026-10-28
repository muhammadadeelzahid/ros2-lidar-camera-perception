#!/usr/bin/env python3
"""Send a goal to the waypoint follower and print its progress.

    ros2 run waypoint_navigation nav_goal forward 5
    ros2 run waypoint_navigation nav_goal turn 90
    ros2 run waypoint_navigation nav_goal waypoints 2,0 2,2 0,2 --relative --speed 0.4
    ros2 run waypoint_navigation nav_goal route route.yaml

forward and turn are relative to where the robot is. Waypoints are x,y pairs in the
odom frame unless --relative is given (x forward, y left of the robot). A route file
holds the same as YAML: waypoints, and optionally relative, speed and heading_deg.
Ctrl-C cancels the goal.
"""

import argparse
import math
import sys
import time

import yaml

import rclpy
from rclpy.action import ActionClient
from rclpy.utilities import remove_ros_args

from geometry_msgs.msg import Point

from perception_msgs.action import FollowWaypoints


def parse(argv):
    ap = argparse.ArgumentParser(prog='nav_goal')
    ap.add_argument('--speed', type=float, default=0.0, help='m/s, 0 for the default')
    sub = ap.add_subparsers(dest='kind', required=True)
    f = sub.add_parser('forward', help='drive straight ahead')
    f.add_argument('metres', type=float)
    t = sub.add_parser('turn', help='turn on the spot, positive is left')
    t.add_argument('degrees', type=float)
    w = sub.add_parser('waypoints', help='drive through x,y points')
    w.add_argument('points', nargs='+')
    w.add_argument('--relative', action='store_true')
    w.add_argument('--heading', type=float, help='final heading, degrees')
    r = sub.add_parser('route', help='waypoints from a YAML file')
    r.add_argument('file')
    return ap.parse_args(argv)


def build_goal(args):
    goal = FollowWaypoints.Goal()
    goal.speed = args.speed
    goal.final_heading = math.nan
    if args.kind == 'forward':
        goal.relative = True
        goal.waypoints = [Point(x=args.metres)]
    elif args.kind == 'turn':
        goal.relative = True
        goal.final_heading = math.radians(args.degrees)
    elif args.kind == 'waypoints':
        goal.relative = args.relative
        for p in args.points:
            x, y = (float(v) for v in p.split(','))
            goal.waypoints.append(Point(x=x, y=y))
        if args.heading is not None:
            goal.final_heading = math.radians(args.heading)
    else:
        with open(args.file) as fh:
            route = yaml.safe_load(fh)
        goal.relative = bool(route.get('relative', False))
        goal.waypoints = [Point(x=float(x), y=float(y)) for x, y in route['waypoints']]
        if 'heading_deg' in route:
            goal.final_heading = math.radians(route['heading_deg'])
        if not args.speed and 'speed' in route:
            goal.speed = float(route['speed'])
    return goal


def main(argv=None):
    args = parse(remove_ros_args(sys.argv if argv is None else argv)[1:])
    rclpy.init(args=argv)
    node = rclpy.create_node('nav_goal')
    client = ActionClient(node, FollowWaypoints, 'follow_waypoints')
    if not client.wait_for_server(timeout_sec=10.0):
        print('waypoint follower not running')
        return 1
    last = [0.0]

    def on_feedback(msg):
        fb = msg.feedback
        if time.time() - last[0] >= 1.0:
            last[0] = time.time()
            line = f'{fb.state:8s} waypoint {fb.waypoint}, {fb.distance_remaining:.1f} m to go'
            print(line + (f'  ({fb.reason})' if fb.reason else ''), flush=True)

    future = client.send_goal_async(build_goal(args), feedback_callback=on_feedback)
    rclpy.spin_until_future_complete(node, future)
    handle = future.result()
    if not handle.accepted:
        print('goal rejected')
        return 1
    result_future = handle.get_result_async()
    try:
        rclpy.spin_until_future_complete(node, result_future)
    except KeyboardInterrupt:
        print('cancelling')
        cancel = handle.cancel_goal_async()
        rclpy.spin_until_future_complete(node, cancel, timeout_sec=3.0)
        return 1
    result = result_future.result().result
    print(f'{"done" if result.success else "failed"}: {result.message}, '
          f'{result.distance_travelled:.1f} m travelled')
    node.destroy_node()
    rclpy.shutdown()
    return 0 if result.success else 1


if __name__ == '__main__':
    sys.exit(main())
