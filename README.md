# ros2-lidar-camera-perception

Camera and 3D LiDAR perception for collision avoidance on a ground vehicle. Objects
are detected in the point cloud, their class is checked against a camera detector,
and the fused detections are tracked over time. A collision monitor turns the tracks
into distance and time-to-collision warnings and can stop the vehicle.

The same pipeline runs on a physical robot or in a Gazebo warehouse with moving
pedestrians and forklifts. In simulation the ground truth is used to label LiDAR
frames automatically for training and evaluating the detector.

## Packages

| Package | Role |
|---|---|
| `lidar_object_detection` | 3D object detection on the point cloud |
| `camera_detection` | 2D detection on the camera image |
| `fusion_tracking` | Camera-LiDAR class confirmation and multi-object tracking |
| `collision_monitor` | Distance / TTC warnings and velocity gate |
| `perception_msgs` | Tracked object and collision warning messages |
| `perception_bringup` | Launch files and configuration for robot and simulation |
| `perception_sim` | Gazebo warehouse and robot model |
| `perception_tools` | Auto-labeling, dataset export, training, calibration, evaluation |

## Pipeline

```
/points             -> lidar_detector     -> /detections/lidar
/camera/image_raw   -> camera_detector    -> /detections/camera
/detections/*       -> fusion_tracker     -> /tracks
/tracks, /odom      -> collision_monitor  -> /collision_warnings
cmd_vel_in          -> collision_monitor  -> cmd_vel
```

Tracks are kept in the `odom` frame so that object velocities are independent of the
vehicle's own motion.

The collision monitor also checks a protective field on the raw point cloud: if
points fall inside a box ahead of the vehicle, whose length grows with speed, it
stops even when nothing was detected. Reverse commands are still allowed so the
vehicle can back away.

## Build

```bash
cd docker
docker compose build
docker compose run --rm dev
# inside the container
colcon build --symlink-install
source install/setup.bash
```

## Simulation

```bash
ros2 launch perception_bringup sim.launch.py
```

Pedestrians and forklifts are kinematic models moved along the paths in
`perception_sim/worlds/warehouse.agents.yaml` by `agent_mover`. Gazebo actors are not
used because ray sensors do not detect them.

Drive with the keyboard from a second terminal. Commands go through the collision
monitor, so the robot stops if a pedestrian or forklift gets too close.

```bash
./scripts/drive.sh
```

To run in a randomly generated warehouse instead of the default one (each world
comes with a matching `.agents.yaml`):

```bash
./scripts/generate_scenarios.sh 20
ros2 launch perception_bringup sim.launch.py world:=$PWD/worlds/scene_004.world
```

## Physical robot

The robot base is expected to publish `/odom` and the `odom -> base_link` transform
and to accept `/cmd_vel`. Teleop or a planner should publish to `cmd_vel_in`.

1. Measure where the LiDAR and camera are mounted and set the offsets in
   `perception_bringup/config/robot_sensors.yaml`.
2. Calibrate the camera (see below) and point `camera_info_url` at the result.
3. Launch:

```bash
ros2 launch perception_bringup robot.launch.py lidar:=velodyne camera:=usb
```

Supported drivers are Velodyne (`lidar_model:=VLP16`, `VLP32C`, ...) for the LiDAR
and usb_cam or RealSense for the camera. For other sensors run their driver
separately and pass `lidar:=none points_topic:=...` or `camera:=none`. Without odometry,
set `tracking_frame` to the base frame and `use_odom: false` in
`perception_bringup/config/perception.yaml`. Set `base_frame`, `ego_length` and
`ego_width` there to match the vehicle so the protective field covers its footprint.

## Calibration

Camera intrinsics from checkerboard images:

```bash
ros2 run perception_tools calibrate_camera --images 'calib/*.png' \
  --rows 7 --cols 9 --square 0.025 \
  --output src/perception_bringup/config/camera_intrinsics.yaml
```

Camera-LiDAR extrinsics: run the overlay node with the sensors live and view
`/calibration/overlay` in RViz or rqt_image_view. Adjust `robot_sensors.yaml` until
the projected points line up with object edges in the image.

```bash
ros2 run perception_tools lidar_camera_overlay
```

## Detector training

Without a trained model the LiDAR detector uses Euclidean clustering. In the default
warehouse it finds about 60% of visible objects within 15 m, with 0.15 m centre error
and 0.85 false positives per frame, mostly pillars and shelf ends. To train a
PointPillars model with OpenPCDet:

```bash
# record several scenarios, one generated world per recording
ros2 launch perception_bringup sim.launch.py world:=$PWD/worlds/scene_001.world
ros2 run perception_tools wander --ros-args -p use_sim_time:=true
./scripts/record_dataset.sh scene_001

# merge them; whole scenarios are assigned to train, val and test
ros2 run perception_tools dataset_export --input dataset --output pcdet_dataset

# train on a CUDA machine with OpenPCDet installed
python3 -m perception_tools.train --data pcdet_dataset \
  --cfg OpenPCDet/tools/cfgs/custom_models/pointpillar.yaml \
  --pretrained pointpillar_7728.pth
```

Then set `openpcdet_cfg` and `checkpoint` in
`lidar_object_detection/config/pointpillars.yaml`. Classes from a pretrained model
are mapped onto ours through `class_aliases`.

A model trained only on simulated data will not transfer directly to a real LiDAR.
Fine-tune and evaluate on a labeled set recorded with the target sensor.

## Evaluation

With the simulation running:

```bash
ros2 run perception_tools eval_metrics --ros-args -p use_sim_time:=true
```

Ctrl-C prints recall, precision, position error and ID switches, and writes
`analysis_output/error_vs_range.png`. Objects with fewer than 5 LiDAR returns are
treated as occluded and not counted; the auto-labeler skips them for the same reason.

## Tests

```bash
colcon test
colcon test-result --verbose
```

## License

MIT. See [LICENSE](LICENSE).
