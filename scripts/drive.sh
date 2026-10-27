#!/bin/bash
# Manual teleop for the robot in simulation. Commands are routed through the collision
# monitor's safety gate (cmd_vel_in -> cmd_vel), so a critical object stops the robot.
exec ros2 run teleop_twist_keyboard teleop_twist_keyboard \
  --ros-args -p use_sim_time:=true -r cmd_vel:=cmd_vel_in
