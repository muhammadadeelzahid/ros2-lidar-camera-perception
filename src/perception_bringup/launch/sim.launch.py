"""Gazebo warehouse, the robot, the perception pipeline and RViz."""

import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    LogInfo,
    OpaqueFunction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def agent_mover(context, *args, **kwargs):
    # Paths for the moving objects live next to the world: foo.world -> foo.agents.yaml
    world = LaunchConfiguration('world').perform(context)
    agents = LaunchConfiguration('agents').perform(context)
    if not agents:
        agents = os.path.splitext(world)[0] + '.agents.yaml'
    if not os.path.exists(agents):
        return [LogInfo(msg=f'no agents file at {agents}; objects will not move')]
    return [Node(
        package='perception_tools', executable='agent_mover', name='agent_mover',
        output='screen',
        parameters=[{'agents_file': agents, 'use_sim_time': True}])]


def generate_launch_description():
    bringup = get_package_share_directory('perception_bringup')
    sim_share = get_package_share_directory('perception_sim')

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(sim_share, 'launch', 'gazebo.launch.py')),
        launch_arguments={
            'use_sim_time': 'true',
            'world': LaunchConfiguration('world'),
            'gui': LaunchConfiguration('gui'),
            'x': LaunchConfiguration('x'),
            'y': LaunchConfiguration('y'),
            'z': LaunchConfiguration('z'),
            'roll': LaunchConfiguration('roll'),
            'pitch': LaunchConfiguration('pitch'),
            'yaw': LaunchConfiguration('yaw'),
            'payload_height': LaunchConfiguration('payload_height'),
            'payload_mass': LaunchConfiguration('payload_mass'),
        }.items(),
    )

    perception = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(bringup, 'launch', 'perception.launch.py')),
        launch_arguments={
            'use_sim_time': 'true',
            'detector_config': LaunchConfiguration('detector_config'),
            'params_file': LaunchConfiguration('params_file'),
        }.items(),
    )

    rviz = Node(
        package='rviz2', executable='rviz2', name='rviz2',
        arguments=['-d', os.path.join(bringup, 'rviz', 'perception.rviz')],
        parameters=[{'use_sim_time': True}],
        condition=IfCondition(LaunchConfiguration('rviz')))

    return LaunchDescription([
        DeclareLaunchArgument(
            'world',
            default_value=PathJoinSubstitution(
                [FindPackageShare('perception_sim'), 'worlds', 'warehouse.world'])),
        DeclareLaunchArgument('agents', default_value='',
                              description='agent paths, defaults to <world>.agents.yaml'),
        DeclareLaunchArgument(
            'detector_config',
            default_value=PathJoinSubstitution(
                [FindPackageShare('lidar_object_detection'), 'config', 'pointpillars.yaml'])),
        DeclareLaunchArgument('gui', default_value='true'),
        DeclareLaunchArgument('rviz', default_value='true'),
        DeclareLaunchArgument('x', default_value='-8.0'),
        DeclareLaunchArgument('y', default_value='0.0'),
        DeclareLaunchArgument('z', default_value='0.1'),
        DeclareLaunchArgument('roll', default_value='0.0'),
        DeclareLaunchArgument('pitch', default_value='0.0'),
        DeclareLaunchArgument('yaw', default_value='0.0'),
        DeclareLaunchArgument('payload_height', default_value='0.0',
                              description='height of an extra mass above the ground, 0 for none'),
        DeclareLaunchArgument('payload_mass', default_value='6.0'),
        DeclareLaunchArgument(
            'params_file',
            default_value=PathJoinSubstitution(
                [FindPackageShare('perception_bringup'), 'config', 'perception.yaml'])),
        gazebo,
        OpaqueFunction(function=agent_mover),
        perception,
        rviz,
    ])
