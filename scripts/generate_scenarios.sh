#!/bin/bash
# Generate N randomized warehouse worlds into worlds/.
#   ./scripts/generate_scenarios.sh 20
N="${1:-10}"
mkdir -p worlds
for i in $(seq 1 "$N"); do
  ros2 run perception_tools generate_world --seed "$i" \
    --output "worlds/scene_$(printf '%03d' "$i").world"
done
