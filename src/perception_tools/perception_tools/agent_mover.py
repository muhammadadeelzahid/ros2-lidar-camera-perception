#!/usr/bin/env python3
"""Move pedestrians and vehicles in Gazebo along scripted paths.

Gazebo actors are not seen by ray sensors, so moving objects are ordinary
kinematic models positioned every cycle through /gazebo/set_entity_state.
Paths come from an agents YAML file next to the world:

    agents:
      - name: pedestrian_1
        speed: 1.2
        mode: pingpong          # or cycle
        waypoints: [[-7.0, 1.5], [7.0, 1.5]]
"""

import math

import yaml

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node

from gazebo_msgs.msg import EntityState
from gazebo_msgs.srv import SetEntityState

from perception_tools.trajectory import pose_at


class AgentMover(Node):

    def __init__(self):
        super().__init__('agent_mover')
        path = self.declare_parameter('agents_file', '').value
        rate = self.declare_parameter('rate', 30.0).value
        with open(path) as fh:
            self.agents = (yaml.safe_load(fh) or {}).get('agents', [])
        self.client = self.create_client(SetEntityState, '/gazebo/set_entity_state')
        self.start = None
        self.create_timer(1.0 / rate, self.tick)
        self.get_logger().info(f'moving {len(self.agents)} agents from {path}')

    def tick(self):
        if not self.client.service_is_ready():
            return
        now = self.get_clock().now().nanoseconds * 1e-9
        if self.start is None:
            self.start = now
        t = now - self.start
        for agent in self.agents:
            x, y, yaw, vx, vy = pose_at(agent['waypoints'], agent['speed'], t,
                                        agent.get('mode', 'pingpong'))
            state = EntityState()
            state.name = agent['name']
            state.reference_frame = 'world'
            state.pose.position.x = x
            state.pose.position.y = y
            state.pose.orientation.z = math.sin(yaw / 2.0)
            state.pose.orientation.w = math.cos(yaw / 2.0)
            state.twist.linear.x = vx
            state.twist.linear.y = vy
            req = SetEntityState.Request()
            req.state = state
            self.client.call_async(req)


def main(argv=None):
    rclpy.init(args=argv)
    node = AgentMover()
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
