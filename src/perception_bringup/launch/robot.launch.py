"""Physical robot: sensor drivers, sensor extrinsics, perception pipeline.

The robot base driver is expected to publish /odom and the odom -> base_link
transform, and to consume /cmd_vel. Teleop or a planner should publish to
cmd_vel_in so commands pass through the collision monitor.
"""

import math
import os

import yaml

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def static_tf(parent, child, xyz, rpy):
    return Node(
        package='tf2_ros', executable='static_transform_publisher',
        name='tf_' + child, output='log',
        arguments=[
            '--x', str(xyz[0]), '--y', str(xyz[1]), '--z', str(xyz[2]),
            '--roll', str(rpy[0]), '--pitch', str(rpy[1]), '--yaw', str(rpy[2]),
            '--frame-id', parent, '--child-frame-id', child,
        ])


def launch_setup(context, *args, **kwargs):
    bringup = get_package_share_directory('perception_bringup')

    def arg(name):
        return LaunchConfiguration(name).perform(context)

    with open(arg('sensors_file')) as fh:
        sensors = yaml.safe_load(fh)
    base = sensors['base_frame']
    lidar_cfg = sensors['lidar']
    cam_cfg = sensors['camera']

    nodes = [
        static_tf(base, lidar_cfg['frame'], lidar_cfg['xyz'], lidar_cfg['rpy']),
        static_tf(base, cam_cfg['frame'], cam_cfg['xyz'], cam_cfg['rpy']),
        # Optical frame convention: z forward, x right, y down.
        static_tf(cam_cfg['frame'], 'camera_optical_frame', [0.0, 0.0, 0.0],
                  [-math.pi / 2, 0.0, -math.pi / 2]),
    ]

    lidar = arg('lidar')
    points_topic = arg('points_topic')
    if lidar == 'velodyne':
        nodes.append(Node(
            package='velodyne_driver', executable='velodyne_driver_node',
            name='velodyne_driver', output='screen',
            parameters=[{
                'device_ip': arg('lidar_ip'),
                'model': arg('lidar_model'),
                'frame_id': lidar_cfg['frame'],
                'rpm': 600.0,
            }]))
        nodes.append(Node(
            package='velodyne_pointcloud', executable='velodyne_transform_node',
            name='velodyne_transform', output='screen',
            parameters=[{
                'model': arg('lidar_model'),
                'calibration': PathJoinSubstitution([
                    FindPackageShare('velodyne_pointcloud'), 'params',
                    arg('lidar_model') + 'db.yaml']).perform(context),
                'min_range': 0.4,
                'max_range': 100.0,
                'organize_cloud': False,
            }],
            remappings=[('velodyne_points', points_topic)]))

    camera = arg('camera')
    image_topic = arg('image_topic')
    info_topic = arg('camera_info_topic')
    use_camera = 'false' if camera == 'none' else 'true'
    if camera == 'usb':
        nodes.append(Node(
            package='usb_cam', executable='usb_cam_node_exe',
            name='usb_cam', output='screen',
            parameters=[{
                'video_device': arg('video_device'),
                'image_width': 640,
                'image_height': 480,
                'framerate': 30.0,
                'frame_id': 'camera_optical_frame',
                'camera_name': 'camera',
                'camera_info_url': arg('camera_info_url'),
            }],
            remappings=[('image_raw', image_topic), ('camera_info', info_topic)]))
    elif camera == 'realsense':
        image_topic = '/camera/color/image_raw'
        info_topic = '/camera/color/camera_info'
        nodes.append(Node(
            package='realsense2_camera', executable='realsense2_camera_node',
            name='camera', namespace='camera', output='screen',
            parameters=[{'enable_depth': False,
                         'rgb_camera.profile': '640x480x30'}]))

    nodes.append(IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(bringup, 'launch', 'perception.launch.py')),
        launch_arguments={
            'use_sim_time': 'false',
            'points_topic': points_topic,
            'image_topic': image_topic,
            'camera_info_topic': info_topic,
            'odom_topic': arg('odom_topic'),
            'use_camera': use_camera,
        }.items(),
    ))

    if arg('rviz') == 'true':
        nodes.append(Node(
            package='rviz2', executable='rviz2', name='rviz2',
            arguments=['-d', os.path.join(bringup, 'rviz', 'perception.rviz')]))
    return nodes


def generate_launch_description():
    bringup = get_package_share_directory('perception_bringup')
    return LaunchDescription([
        DeclareLaunchArgument(
            'sensors_file',
            default_value=os.path.join(bringup, 'config', 'robot_sensors.yaml'),
            description='sensor mounting positions relative to the base frame'),
        DeclareLaunchArgument('lidar', default_value='velodyne',
                              description='velodyne, or none to use an external driver'),
        DeclareLaunchArgument('lidar_model', default_value='VLP16'),
        DeclareLaunchArgument('lidar_ip', default_value='192.168.1.201'),
        DeclareLaunchArgument('points_topic', default_value='/points'),
        DeclareLaunchArgument('camera', default_value='usb',
                              description='usb, realsense or none'),
        DeclareLaunchArgument('video_device', default_value='/dev/video0'),
        DeclareLaunchArgument(
            'camera_info_url',
            default_value='file://' + os.path.join(
                bringup, 'config', 'camera_intrinsics.yaml')),
        DeclareLaunchArgument('image_topic', default_value='/camera/image_raw'),
        DeclareLaunchArgument('camera_info_topic', default_value='/camera/camera_info'),
        DeclareLaunchArgument('odom_topic', default_value='/odom'),
        DeclareLaunchArgument('rviz', default_value='false'),
        OpaqueFunction(function=launch_setup),
    ])
