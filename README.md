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
| `waypoint_navigation` | Waypoint following with pure pursuit on odometry |

## Pipeline

```
/points             -> lidar_detector     -> /detections/lidar
/camera/image_raw   -> camera_detector    -> /detections/camera
/detections/*       -> fusion_tracker     -> /tracks
/tracks, /odom      -> collision_monitor  -> /collision_warnings
/imu, /odom         -> collision_monitor  -> /stability
waypoint goal       -> waypoint_follower  -> cmd_vel_in
cmd_vel_in          -> collision_monitor  -> cmd_vel
```

Tracks are kept in the `odom` frame so that object velocities are independent of the
vehicle's own motion.

The collision monitor also checks a protective field on the raw point cloud: if
points fall inside a box ahead of the vehicle, whose length grows with speed, it
stops even when nothing was detected. Reverse commands are still allowed so the
vehicle can back away.

With an IMU the monitor also watches for tip-over. A complementary filter estimates
roll and pitch, using odometry to remove the vehicle's own acceleration. The margin
is the force-angle measure: for each edge of the polygon formed by the wheel contacts,
the angle between the net force at the centre of gravity and the line from the
centre of gravity to that edge. Before a command is passed on, the monitor predicts
the margin it would produce, adding the turn's lateral acceleration v * w to gravity
at the current slope, and scales speed and turn rate down until the margin stays
above `stability_min_margin_deg`. Slowing is rate limited so the stop itself does
not tip the vehicle.

## Failsafes

| Failure | Response |
|---|---|
| LiDAR stops, or the detector crashes or is still loading | `/tracks` stops; robot held stopped after `tracks_timeout` (0.5 s) |
| Part of the LiDAR view blank, e.g. dirt or a blocked sensor | speed capped at `degraded_max_speed` (0.3 m/s) |
| Most of the LiDAR view blank | robot held stopped |
| Camera stops | carries on with LiDAR classes; camera detections older than 0.3 s are not used |
| Odometry stops | robot held stopped after `odom_timeout` (0.5 s) |
| IMU stops | robot held stopped after `imu_timeout` (0.2 s) |
| Collision monitor stops | drive stops when the heartbeat or commands go quiet |

The detector judges LiDAR health from the number of returns and how much of the
view they cover: on level ground the floor returns points in every direction, so an
empty sector means something is in the way of the sensor. Each node reports its
state on `/diagnostics`; the monitor's `gate` entry says whether it is passing,
limiting or stopping commands and why. View it with `rqt_robot_monitor`.

The monitor publishes a heartbeat on `collision_monitor/heartbeat`. In simulation
`drive_watchdog` stands between the monitor and the wheels and stops them if the
heartbeat or the commands go quiet. On a real vehicle the drive controller should do
the same, since many drives keep executing the last command they received.

The fault test drives the robot in simulation and switches each failure on in turn,
feeding the perception nodes through relays it can stop or degrade:

```bash
./scripts/fault_test.sh
```

| Fault | Robot | Time to stop |
|---|---|---|
| LiDAR stops | stopped | 1.1 s |
| 135 degrees of LiDAR view blank | slowed from 0.5 to 0.3 m/s | |
| 250 degrees of LiDAR view blank | stopped | 0.6 s |
| Camera stops | kept driving, LiDAR only | |
| Odometry stops | stopped | 1.2 s |
| IMU stops | stopped | 0.7 s |
| Collision monitor killed | stopped by the drive watchdog | 0.6 s |

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

## Navigation

`waypoint_follower` drives to waypoints with pure pursuit on odometry, through the
collision monitor like any other command source. It does not plan around obstacles:
when the monitor stops the robot it waits, reports why, and gives up after
`blocked_timeout` without progress. Goals use the `FollowWaypoints` action; `nav_goal`
sends one from the command line and prints progress. Ctrl-C cancels.

```bash
ros2 run waypoint_navigation nav_goal forward 5
ros2 run waypoint_navigation nav_goal turn 90
ros2 run waypoint_navigation nav_goal waypoints 3,0 3,3 0,3 0,0 --relative --heading 0
ros2 run waypoint_navigation nav_goal route src/perception_bringup/config/warehouse_route.yaml
```

Without `--relative`, waypoints are in the odom frame. In simulation the robot came
back to within 1 cm after a 3 m square, finished 500 m straight at 1 m/s 2 cm from the
goal, and drove the 48 m warehouse loop in 113 s with pedestrians and the forklift
crossing it, held three times by the collision monitor. Simulated odometry is exact;
on a vehicle, wheel odometry drifts with distance and long routes need a localization
source such as LiDAR odometry or GNSS.

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

For tip-over monitoring set `support_points` to the wheel contact points and `cg` to
the centre of gravity, both relative to a base frame on the ground. Raise the `cg`
height when the vehicle carries a load high up. Set the IMU mount in
`robot_sensors.yaml` and calibrate it as below. On a vehicle without an IMU, launch
with `use_imu:=false`; otherwise the monitor holds it stopped while waiting for
`/imu`.

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

IMU mounting: with the vehicle standing still on level ground, record the sensor's
tilt and gyro bias, then pass the file at launch with `imu_calibration:=...`.

```bash
ros2 run perception_tools calibrate_imu --ros-args -p output:=imu_calibration.yaml
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

## Tip-over test

`slopes.world` has side-slope pads at 4, 8 and 12 degrees and a strip of 2 cm bumps.
The test vehicle is the robot with four driven wheels, `wheels:=4`, and a 6 kg mass at
0.9 m, `payload_height:=0.9`. That puts the centre of gravity at about 0.47 m, so it
tips at about 20 degrees sideways. The script drives each scenario three times with
the stability gate off and three times with it on:

```bash
./scripts/stability_scenarios.sh stability_results
```

A run counts as a tip-over when a wheel is more than 10 cm off the surface, from the
Gazebo ground truth. Bouncing and skid-steer chatter lift a wheel up to about 5 cm.

| Scenario | Tipped, gate off | Tipped, gate on | Worst wheel lift, off | Worst wheel lift, on |
|---|---|---|---|---|
| Flat, turn at 1.0 m/s and 2.5 rad/s | 0/3 | 0/3 | 3 cm | 5 cm |
| 4 degree slope, turn at 1.0 m/s and 2.0 rad/s | 0/3 | 0/3 | 2 cm | 3 cm |
| 8 degree slope, turn at 1.2 m/s and 2.4 rad/s | 3/3 | 0/3 | 44 cm | 4 cm |
| 12 degree slope, straight across at 0.6 m/s | 0/3 | 0/3 | 1 cm | 1 cm |
| 12 degree slope, turn at 0.8 m/s and 2.0 rad/s | 3/3 | 0/3 | 45 cm | 1 cm |
| Bumps 0.6 m apart, straight at 0.8 m/s | 1/3 | 0/3 | 42 cm | 10 cm |

Turns on the slopes go towards the uphill side, which puts the outside of the turn
downhill. With the gate off, the first critical warning came at least 2.3 s before a
wheel lifted. The bumps were meant as a false-alarm check, but hitting them at about
1.3 Hz builds up pitching in the tall vehicle and it went over once; with the gate on
it slowed down and stayed up, though one run came close.

Roll and pitch estimates were within about 1 to 2 degrees RMS of the ground truth while
driving and 0.3 degrees standing on the 12 degree slope. `calibrate_imu` measured the
simulated IMU's mounting error as 1.07 and -0.51 degrees; the set values are 1.0 and
-0.6. The protective field assumes a flat floor and is turned off for this test,
because a slope ahead looks like an obstacle to it.

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
