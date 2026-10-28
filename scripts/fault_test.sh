#!/bin/bash
# Sensor fault tests in simulation: the perception nodes read their sensors through
# /test/* relays that fault_test stops or degrades on cue.
#   ./scripts/fault_test.sh [log_file] [extra fault_test arguments]
# Runs on open ground (slopes.world near the origin) so nothing in the way stops the robot.
set -e
LOG="${1:-fault_test_launch.log}"
shift || true

setsid ros2 launch perception_bringup sim.launch.py \
  world:="$(ros2 pkg prefix perception_sim)/share/perception_sim/worlds/slopes.world" \
  x:=0.0 gui:=false rviz:=false \
  points_topic:=/test/points image_topic:=/test/image \
  odom_topic:=/test/odom imu_topic:=/test/imu \
  > "$LOG" 2>&1 &
LAUNCH=$!
trap 'kill -INT -$LAUNCH 2>/dev/null; sleep 5; kill -9 -$LAUNCH 2>/dev/null' EXIT

until ros2 topic list 2>/dev/null | grep -q '^/stability$'; do sleep 2; done
sleep 5
ros2 run perception_tools fault_test --ros-args "$@"
