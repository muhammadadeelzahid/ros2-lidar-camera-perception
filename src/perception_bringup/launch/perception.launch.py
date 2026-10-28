"""Perception nodes shared by the simulation and robot entry points."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time')
    params = LaunchConfiguration('params_file')
    common = [params, {'use_sim_time': use_sim_time}]

    detector = Node(
        package='lidar_object_detection', executable='detector_node',
        name='lidar_detector', output='screen',
        parameters=common + [{'config': LaunchConfiguration('detector_config')}],
        remappings=[('/points', LaunchConfiguration('points_topic'))])

    camera = Node(
        package='camera_detection', executable='yolo_node',
        name='camera_detector', output='screen', parameters=common,
        remappings=[('/camera/image_raw', LaunchConfiguration('image_topic'))],
        condition=IfCondition(LaunchConfiguration('use_camera')))

    tracker = Node(
        package='fusion_tracking', executable='tracker_node',
        name='fusion_tracker', output='screen', parameters=common,
        remappings=[('/camera/camera_info',
                     LaunchConfiguration('camera_info_topic'))])

    monitor = Node(
        package='collision_monitor', executable='monitor_node',
        name='collision_monitor', output='screen',
        parameters=common + [{
            'use_stability': ParameterValue(LaunchConfiguration('use_imu'), value_type=bool),
            'imu_calibration_file': LaunchConfiguration('imu_calibration'),
        }],
        remappings=[('/odom', LaunchConfiguration('odom_topic')),
                    ('/imu', LaunchConfiguration('imu_topic')),
                    ('/points', LaunchConfiguration('points_topic'))])

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument(
            'params_file',
            default_value=PathJoinSubstitution(
                [FindPackageShare('perception_bringup'), 'config', 'perception.yaml'])),
        DeclareLaunchArgument(
            'detector_config',
            default_value=PathJoinSubstitution(
                [FindPackageShare('lidar_object_detection'), 'config', 'pointpillars.yaml']),
            description='detector settings; set openpcdet_cfg and checkpoint to use PointPillars'),
        DeclareLaunchArgument('points_topic', default_value='/points'),
        DeclareLaunchArgument('image_topic', default_value='/camera/image_raw'),
        DeclareLaunchArgument('camera_info_topic', default_value='/camera/camera_info'),
        DeclareLaunchArgument('odom_topic', default_value='/odom'),
        DeclareLaunchArgument('use_camera', default_value='true',
                              description='run the camera detector'),
        DeclareLaunchArgument('use_imu', default_value='true',
                              description='tip-over monitoring; the robot will not move '
                                          'without IMU data while this is true'),
        DeclareLaunchArgument('imu_topic', default_value='/imu'),
        DeclareLaunchArgument('imu_calibration', default_value='',
                              description='YAML written by calibrate_imu'),
        detector,
        camera,
        tracker,
        monitor,
    ])
