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

Randomly generated warehouses can be used instead of the default one. Each world
comes with a matching `.agents.yaml`.

```bash
./scripts/generate_scenarios.sh 20
ros2 launch perception_bringup sim.launch.py world:=$PWD/worlds/scene_004.world
```

## Physical robot

The robot base is expected to publish `/odom` and the `odom -> base_link` transform
and to accept `/cmd_vel`. Teleop or a planner should publish to `cmd_vel_in`.

1. Measure where the LiDAR and camera are mounted and set the offsets in
   `perception_bringup/config/robot_sensors.yaml`.
2. Calibrate the camera as described below and point `camera_info_url` at the result.
3. Launch:

```bash
ros2 launch perception_bringup robot.launch.py lidar:=velodyne camera:=usb
```

Supported drivers are Velodyne for the LiDAR, set with `lidar_model:=VLP16`,
`VLP32C` and so on, and usb_cam or RealSense for the camera. For other sensors run their driver
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

Without a trained model the LiDAR detector uses Euclidean clustering. To train a
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
  --cfg src/lidar_object_detection/config/pcdet/pointpillar_warehouse.yaml
```

To use the trained model, copy `lidar_object_detection/config/pointpillars.yaml`, set
`openpcdet_cfg` to the training config and `checkpoint` to the trained model, and pass
it at launch. torch and OpenPCDet must be importable by the ROS Python.

```bash
ros2 launch perception_bringup sim.launch.py detector_config:=/path/to/pointpillars_trained.yaml
```

A model trained on another dataset can be used too; its classes are mapped onto ours
through `class_aliases`.

Results on the default warehouse, a layout not used for training. The test set has
1,141 frames, objects within 15 m are counted, and a detection is correct when its
centre is within 1 m.

| Detector | Recall | Precision | False positives / frame | Centre error | Class accuracy |
|---|---|---|---|---|---|
| Clustering | 0.85 | 0.27 | 2.45 | 0.24 m | 0.84 |
| PointPillars, score >= 0.3 | 0.94 | 0.75 | 0.32 | 0.04 m | 0.99 |
| PointPillars, score >= 0.5 | 0.81 | 0.95 | 0.04 | 0.03 m | 1.00 |

The model was trained from scratch on 7,410 frames from 17 generated warehouses for
40 epochs, which took about 45 minutes on one L40S. It runs in about 4 ms per scan on
that GPU. Each generated warehouse has 2 to 5 posts besides the shelves and pallets;
without them the model detected the pillars in the test warehouse as pedestrians.
Most of the remaining false positives are shelf ends detected as vehicles.

Reproduce the numbers with:

```bash
ros2 run perception_tools eval_detector --data dataset/warehouse --config <detector config>
```

These are simulation results. A model trained only on simulated data will not
transfer directly to a real LiDAR; fine-tune and evaluate on a labeled set recorded
with the target sensor.

## Closed-loop test

The full pipeline in the default warehouse, with `wander` driving the robot through
the collision monitor for 150 s while a forklift and two pedestrians move along
their paths:

| Detector | Recall | Precision | Track error | Detection time | Phantom critical warnings | Distance driven |
|---|---|---|---|---|---|---|
| Clustering | 0.79 | 0.28 | 0.44 m | 10 ms | 1267 | 2.4 m |
| PointPillars | 0.91 | 0.82 | 0.13 m | 6 ms | 0 | 37.2 m |

A phantom warning is a critical warning on a track with no real object near it.
With clustering these kept the robot in emergency stop for most of the run. With
PointPillars the stops were all for real objects or the protective field. The robot did
not drive into anything in any of the runs.

The monitor also holds the robot stopped when no tracks have arrived for longer than
`tracks_timeout`, 0.5 s by default. This covers the detector loading its model at
startup or crashing.

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
