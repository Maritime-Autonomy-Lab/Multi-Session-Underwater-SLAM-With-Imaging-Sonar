from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    pkg_share = get_package_share_directory('boat_slam')

    return LaunchDescription([

        # Dead Reckoning Node
        Node(
            package='boat_slam',
            executable='dead_reckoning_lidar',
            name='dead_reckoning_lidar',
            output='screen',
            emulate_tty=True,
            parameters=[os.path.join(pkg_share, 'config', 'dead_reckoning.yaml')],
            env={'PYTHONUNBUFFERED': '1'}
        ),

        # Feature Extraction Node
        Node(
            package='boat_slam',
            executable='feature_extraction',
            name='feature_extraction',
            output='screen',
            emulate_tty=True,
            parameters=[os.path.join(pkg_share, 'config', 'feature_extraction.yaml')],
            env={'PYTHONUNBUFFERED': '1'}
        ),

        # SLAM Node
        Node(
            package='boat_slam',
            executable='object_slam_node',
            name='object_slam_node',
            output='screen',
            emulate_tty=True,
            parameters=[os.path.join(pkg_share, 'config', 'slam.yaml')],
            env={'PYTHONUNBUFFERED': '1'}
        ),

        # RViz
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=['-d', os.path.join(pkg_share, 'config', 'field.rviz')]
        ),
    ])
