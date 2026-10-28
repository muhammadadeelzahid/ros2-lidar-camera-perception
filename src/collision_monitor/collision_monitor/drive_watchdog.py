#!/usr/bin/env python3
"""Drive-side watchdog for the simulated robot.

Stands in for what the vehicle's drive controller should do: pass velocity commands
from the collision monitor to the wheels, and stop the wheels when the commands or the
monitor's heartbeat go quiet. Without it, a drive that holds the last command keeps
the robot moving after the monitor or the planner has died. On a real vehicle this
belongs in the drive controller or ECU, not in another ROS node.

    cmd_vel (from the monitor) + collision_monitor/heartbeat -> drive/cmd_vel
"""

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node

from geometry_msgs.msg import Twist
from std_msgs.msg import UInt32


class DriveWatchdog(Node):

    def __init__(self):
        super().__init__('drive_watchdog')
        self.cmd_timeout = self.declare_parameter('cmd_timeout', 0.5).value
        self.heartbeat_timeout = self.declare_parameter('heartbeat_timeout', 0.3).value
        self.cmd = Twist()
        self.cmd_time = None
        self.beat = None
        self.beat_time = None
        self.stopped_reason = 'waiting for the collision monitor'
        self.create_subscription(Twist, 'cmd_vel', self.on_cmd, 10)
        self.create_subscription(UInt32, 'collision_monitor/heartbeat', self.on_beat, 10)
        self.pub = self.create_publisher(Twist, 'drive/cmd_vel', 10)
        self.create_timer(0.02, self.on_timer)

    def _age(self, t):
        return float('inf') if t is None else (self.get_clock().now() - t).nanoseconds * 1e-9

    def on_cmd(self, msg):
        self.cmd = msg
        self.cmd_time = self.get_clock().now()

    def on_beat(self, msg):
        # A heartbeat that stops counting means the monitor is stuck, not alive.
        if msg.data != self.beat:
            self.beat = msg.data
            self.beat_time = self.get_clock().now()

    def on_timer(self):
        if self._age(self.beat_time) > self.heartbeat_timeout:
            reason = 'no heartbeat from the collision monitor'
        elif self._age(self.cmd_time) > self.cmd_timeout:
            reason = 'no velocity commands'
        else:
            reason = ''
        if reason != self.stopped_reason:
            if reason:
                self.get_logger().warn(f'{reason}; stopping the wheels')
            else:
                self.get_logger().info('commands and heartbeat present; driving')
            self.stopped_reason = reason
        self.pub.publish(Twist() if reason else self.cmd)


def main(argv=None):
    rclpy.init(args=argv)
    node = DriveWatchdog()
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
