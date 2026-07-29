import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    GroupAction,
    IncludeLaunchDescription,
    SetEnvironmentVariable,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.descriptions import ParameterFile

def generate_launch_description():
    return LaunchDescription([
        # Log message to indicate launch
        # LogInfo(msg="Starting Dead Reckoning and Lidar Pre-Processing Nodes"),
        


        # Include the kiss_icp launch file (with an argument for the topic)
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(get_package_share_directory('kiss_icp'), 'launch', 'odometry.launch.py')
            ),
            launch_arguments={'topic': '/ouster/xyz_cloud'}.items()
        )
    ])
