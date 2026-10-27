import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    SetEnvironmentVariable,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    sim_share = get_package_share_directory('perception_sim')
    gazebo_ros_share = get_package_share_directory('gazebo_ros')

    use_sim_time = LaunchConfiguration('use_sim_time')
    world = LaunchConfiguration('world')
    x = LaunchConfiguration('x')
    y = LaunchConfiguration('y')
    z = LaunchConfiguration('z')
    yaw = LaunchConfiguration('yaw')

    xacro_file = os.path.join(sim_share, 'urdf', 'robot.urdf.xacro')
    robot_description = ParameterValue(
        Command(['xacro ', xacro_file,
                 ' payload_height:=', LaunchConfiguration('payload_height'),
                 ' payload_mass:=', LaunchConfiguration('payload_mass')]),
        value_type=str)

    model_path = os.path.join(sim_share, 'models')
    existing = os.environ.get('GAZEBO_MODEL_PATH', '')
    set_model_path = SetEnvironmentVariable(
        'GAZEBO_MODEL_PATH',
        model_path if not existing else model_path + os.pathsep + existing)

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gazebo_ros_share, 'launch', 'gazebo.launch.py')),
        launch_arguments={
            'world': world,
            'verbose': 'false',
            'gui': LaunchConfiguration('gui'),
        }.items(),
    )

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': robot_description,
            'use_sim_time': use_sim_time,
        }],
    )

    spawn = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        output='screen',
        arguments=[
            '-topic', 'robot_description',
            '-entity', 'ego_bot',
            '-x', x, '-y', y, '-z', z,
            '-R', LaunchConfiguration('roll'), '-P', LaunchConfiguration('pitch'), '-Y', yaw,
        ],
    )

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('gui', default_value='true',
                              description='start the Gazebo client'),
        DeclareLaunchArgument(
            'world',
            default_value=PathJoinSubstitution(
                [FindPackageShare('perception_sim'),
                 'worlds', 'warehouse.world'])),
        DeclareLaunchArgument('x', default_value='-8.0'),
        DeclareLaunchArgument('y', default_value='0.0'),
        DeclareLaunchArgument('z', default_value='0.1'),
        DeclareLaunchArgument('roll', default_value='0.0'),
        DeclareLaunchArgument('pitch', default_value='0.0'),
        DeclareLaunchArgument('yaw', default_value='0.0'),
        DeclareLaunchArgument('payload_height', default_value='0.0',
                              description='height of an extra mass above the ground, 0 for none'),
        DeclareLaunchArgument('payload_mass', default_value='6.0'),
        set_model_path,
        gazebo,
        robot_state_publisher,
        spawn,
    ])
