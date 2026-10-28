#!/bin/bash
# Tip-over scenarios in slopes.world, stability gate off then on. The robot is the
# four-wheel variant with a 6 kg mass at 0.9 m, which puts its CG at about 0.47 m.
#   ./scripts/stability_scenarios.sh [output_dir] [extra runner arguments]
# Collision stops and the protective field are turned off so that only the stability
# gate changes commands. The protective field assumes a flat floor and would stop the
# robot whenever it faces up a slope.
set -e
OUT="${1:-stability_results}"
shift || true
SHARE="$(ros2 pkg prefix perception_bringup)/share/perception_bringup"
PARAMS="$(mktemp --suffix=.yaml)"

python3 - "$SHARE/config/perception.yaml" "$PARAMS" <<'EOF'
import sys
import yaml
cfg = yaml.safe_load(open(sys.argv[1]))
mon = cfg['/collision_monitor']['ros__parameters']
mon.update(enable_estop=False, use_protective_field=False, cg=[0.0, 0.0, 0.47],
           support_points=[0.14, 0.17, 0.14, -0.17, -0.14, 0.17, -0.14, -0.17])
yaml.safe_dump(cfg, open(sys.argv[2], 'w'))
EOF

setsid ros2 launch perception_bringup sim.launch.py \
  world:="$(ros2 pkg prefix perception_sim)/share/perception_sim/worlds/slopes.world" \
  x:=0.0 wheels:=4 payload_height:=0.9 params_file:="$PARAMS" gui:=false rviz:=false \
  > "${OUT}_launch.log" 2>&1 &
LAUNCH=$!
trap 'kill -INT -$LAUNCH 2>/dev/null; sleep 5; kill -9 -$LAUNCH 2>/dev/null; rm -f "$PARAMS"' EXIT

until ros2 topic list 2>/dev/null | grep -q '^/stability$'; do sleep 2; done
sleep 5
ros2 run perception_tools stability_scenarios --ros-args -p output_dir:="$OUT" "$@"
