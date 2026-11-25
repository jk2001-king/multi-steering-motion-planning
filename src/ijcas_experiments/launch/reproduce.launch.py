from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('output',default_value='',description='New output directory; empty selects a timestamped results folder'),
        ExecuteProcess(cmd=['ros2','run','ijcas_experiments','reproduce_results',
            '--output',LaunchConfiguration('output')],output='screen'),
    ])
