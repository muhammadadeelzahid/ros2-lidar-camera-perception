#!/bin/bash
# Record one auto-labeled scenario while the simulation runs.
#   ./scripts/record_dataset.sh scene_007
# Writes dataset/<name>/points and dataset/<name>/labels. Ctrl-C to stop.
NAME="${1:-default}"
exec ros2 run perception_tools auto_labeler \
  --ros-args -p use_sim_time:=true -p out_dir:="dataset/${NAME}"
