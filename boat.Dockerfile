ARG ROS_DISTRO=humble
FROM ros:${ROS_DISTRO}

ARG DEBIAN_FRONTEND=noninteractive 

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update && apt-get install --no-install-recommends -y \
    usbutils \
    nano \
    python3-colcon-common-extensions \
    python3-colcon-clean \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN sudo apt update
RUN sudo apt-get install build-essential
RUN sudo apt update

RUN sudo apt-get install usbutils
RUN sudo apt-get install -y libpcap-dev 
RUN sudo apt install ros-humble-diagnostic-updater
RUN sudo apt-get install -y libpcap-dev 
RUN sudo apt-get install -y ros-humble-angles
RUN sudo apt-get install -y ros-humble-pcl-ros
RUN sudo apt-get install -y ros-humble-cv-bridge
RUN sudo apt-get install -y ros-humble-cv-bridge
RUN sudo apt install -y ros-humble-rmw-cyclonedds-cpp
# RUN sudo apt-get install -y ros-humble-image-transport
#RUN sudo apt-get install usbutils

# RUN sudo apt install python3-colcon-common-extension
RUN sudo apt install -y python3-pip
RUN pip install pymavlink
RUN pip install pyserial
RUN pip install mavproxy
RUN pip install smbus2
RUN pip install websocket-client

# RUN pip install cv_bridge==4.1.0
# RUN pip install geometry_msgs==5.3.6
RUN pip install gtsam
RUN pip install matplotlib==3.6.3
#RUN pip install message_filters==4.11.5
#RUN pip install nav_msgs==5.3.6
#RUN pip install numpy==2.4.2
RUN pip install opencv_python==4.11.0.86
#RUN pip install rclpy==7.1.4
RUN pip install scikit_learn==1.4.1.post1
# RUN pip install scipy==1.17.1
# RUN pip install sensor_msgs==5.3.6
# RUN pip install sensor_msgs_py==5.3.6
RUN pip install Shapely==2.1.2
# RUN pip install std_msgs==5.3.6
RUN pip install supervision==0.27.0.post1
# RUN pip install tf2_ros_py==0.36.9
RUN pip install ultralytics==8.3.166
# RUN pip install visualization_msgs==5.3.6

USER 0
RUN mkdir -p /ros_ws/src/
USER $CONTAINER_USER_ID

# copy the startup script
COPY start_in_docker.sh start_in_docker.sh
COPY warm_up.sh warm_up.sh
RUN sudo chmod +x ./start_in_docker.sh
RUN sudo chmod +x ./warm_up.sh

# copy some utils scripts
COPY boat_packages/bash_scripts/check_hz.sh check_hz.sh 
COPY boat_packages/bash_scripts/check_data.sh check_data.sh 
RUN sudo chmod +x ./check_hz.sh
RUN sudo chmod +x ./check_data.sh

# copy the code
COPY boat_packages/vectornav /ros_ws/src/vectornav
COPY boat_packages/starfish_ros /ros_ws/src/starfish_ros
COPY boat_packages/velodyne /ros_ws/src/velodyne
COPY boat_packages/dvl_a50 /ros_ws/src/dvl_a50
COPY boat_packages/dvl_msgs /ros_ws/src/dvl_msgs
COPY boat_packages/ouster-ros /ros_ws/src/ouster-ros
COPY boat_packages/sonar_oculus /ros_ws/src/sonar_oculus
COPY /libnabo /ros_ws/src/libnabo
COPY /libpointmatcher /ros_ws/src/libpointmatcher

RUN apt-get update && apt-get install -y python3-dev python3.10-dev libpython3.10-dev

RUN sudo apt-get install -y ros-humble-image-transport
RUN sudo apt-get install -y ros-humble-sensor-msgs-py

#build
#RUN /ros_entrypoint.sh colcon build --base-paths ros_ws/ --build-base ros_ws/build --install-base ros_ws/install
RUN /ros_entrypoint.sh
#RUN source /opt/ros/humble/setup.bash 
WORKDIR ros_ws/
#RUN colcon build
RUN /bin/bash -c "source /opt/ros/humble/setup.bash; colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release"
# RUN sed -i "$(wc -l < /ros_entrypoint.sh)i\\source \"//install/setup.bash\"\\" /ros_entrypoint.sh

# COPY /Acoustic-Modems /ros_ws/Acoustic-Modems
# COPY /sonar-slam-ros2 /ros_ws/src/sonar-slam-ros2
COPY /boat_slam /ros_ws/src/boat_slam
RUN /bin/bash -c "source /opt/ros/humble/setup.bash; colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release"


ENTRYPOINT [ "/ros_entrypoint.sh" ]