from setuptools import setup
import os
from glob import glob

package_name = 'boat_slam'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        # Package XML and resource
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Config files
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        # Launch files
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    author='Jake',
    author_email='you@example.com',
    description='Boat SLAM Python nodes',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'dead_reckoning_lidar = boat_slam.dead_reckoning_lidar:main',
            'feature_extraction = boat_slam.feature_extraction:main',
            'object_slam_node = boat_slam.object_slam_node:main',
        ],
    },
)
