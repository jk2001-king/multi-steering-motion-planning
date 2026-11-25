from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('configuration',description='Explicit new checked bridge configuration'),
        ExecuteProcess(cmd=['ros2','run','ijcas_experiments','sensor_bridge',
            LaunchConfiguration('configuration')],output='screen'),
    ])
