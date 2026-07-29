# Multi-Session-Underwater-SLAM-With-Imaging-Sonar
This repo is a multi-session SLAM system based on objects, rather than ICP or image processing. If you find this hepful, please cite us! 

The views expressed in this repo are those of the author(s) and do not reflect the official policy or position of the U.S. Naval Academy, Department of the Navy, the Department of Defense, or the U.S. Government.


# Sensor overview

Our vehicle is documented in this repo: Link coming soon! 
- Occulus M750d imaging sonar
- Occulus M1200 imaging sonar (optional)
- Rowe SeaPilot DVL
- Vectornav 100 MEMS IMU
- Bar30 pressure sensor
- KVH-DSP-1760 fiber optic gyroscope (optional)

# Python Dependencies, note python-3

See docker image

# Installation
- Ensure all python dependencies are installed
- Check ros distro
- clone this repo into your workspace
- clone git clone https://github.com/ethz-asl/libnabo.git into your workspace
- clone https://github.com/ethz-asl/libpointmatcher.git into your workspace
- clone hhttps://github.com/jake3991/boat_docker_with_packages.git into your workspace
- colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release

# Sample data
We provide a rosbag data file to test and run the SLAM system. Available here: link coming soon! 

# Running "Online"
This will launch the SLAM system, then we will playback the data as if it is happening now. 
- source workspace/install/setup.bash
- ros2 launch boat_slam slam.xml
- ros2 bag play your_data.bag


# Current To Do list
- enhance some of the cpp documentation for CFAR

# Citation
If you use this repo please cite the following work. Link to pre-print here: https://arxiv.org/abs/2202.08359

```
@ARTICLE{11520259,
  author={McConnell, John and Huang, Yewei and Morris, Thomas and Doughty, Josh and Moynihan, Dennis},
  journal={IEEE Robotics and Automation Letters}, 
  title={Multi-Session SLAM for Imaging Sonar Equipped Underwater Vehicles Using Semantic Scene Graphs}, 
  year={2026},
  volume={11},
  number={7},
  pages={8092-8099},
  keywords={Simultaneous localization and mapping;Sonar;Robots;Signal detection;YOLO;Bridges;Runtime;Standards;Trajectory;Current;Marine robotics;SLAM;range sensing},
  doi={10.1109/LRA.2026.3693583}}

```