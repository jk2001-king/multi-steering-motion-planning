"""One selector entry. The default checks settings and never starts motion."""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    names = ('config', 'environment', 'scenario', 'method', 'condition', 'feedback',
             'stage', 'output', 'sdk_python', 'headless', 'allow_motion')
    arguments = []
    for name in names:
        arguments.extend(['--' + name.replace('_', '-'), LaunchConfiguration(name)])
    return LaunchDescription([
        *[DeclareLaunchArgument(name, default_value='',
            description='Override the experiment YAML (' + name + ')') for name in names],
        Node(package='ijcas_experiments', executable='experiment', name='ijcas_experiment',
             arguments=arguments, output='screen'),
    ])
