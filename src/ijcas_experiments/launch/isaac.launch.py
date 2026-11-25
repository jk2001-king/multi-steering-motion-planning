from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('executable',default_value='isaacsim'),
        ExecuteProcess(cmd=['ros2','run','ijcas_experiments','isaac',
            '--executable',LaunchConfiguration('executable')],output='screen'),
    ])
