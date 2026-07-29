cmake_minimum_required(VERSION 3.5)
project(boat_slam)

set(CMAKE_CXX_STANDARD 17)
set( CMAKE_CXX_FLAGS " -pthread " )

# Find packages in ROS 2
find_package(pybind11 REQUIRED)
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(std_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(sonar_oculus REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(OpenCV REQUIRED)
find_package(libpointmatcher REQUIRED)
find_package(PCL REQUIRED)
find_package(sensor_msgs REQUIRED)


include_directories(${libpointmatcher_INCLUDE_DIRS})
include_directories(${sonar_oculus_INCLUDE_DIRS})


pybind11_add_module(cfar src/cpp/cfar.cpp)
target_link_libraries(cfar PRIVATE pybind11::module)

pybind11_add_module(pcl src/cpp/pcl.cpp)
target_link_libraries(pcl PRIVATE pybind11::module)

# set_target_properties(cfar PROPERTIES
#  LIBRARY_OUTPUT_DIRECTORY lib/${PROJECT_NAME}}
# )

# Add directories to include
include_directories(
  include
  ${rclcpp_INCLUDE_DIRS}
  ${cv_bridge_INCLUDE_DIRS}
  ${image_transport_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

#add_executable(feature_extraction_node
#  src/feature_extraction_node.cpp
#)

#rosidl_generate_interfaces(${PROJECT_NAME}
#  "msg/DeadReckoning.msg"
#  DEPENDENCIES std_msgs sensor_msgs
# )
 
# Link relevant external dependencies
#ament_target_dependencies(
  #feature_extraction_node
    #rclcpp
    #cv_bridge
    #sonar_oculus
#)

# Link libraries to the executable target
#target_link_libraries(feature_extraction_node
#  ${rclcpp_LIBRARIES}
#  ${sensor_msgs_LIBRARIES}
#  ${cv_bridge_LIBRARIES}
#  ${image_transport_LIBRARIES}
#  ${OpenCV_LIBRARIES}
#)

# Install the header files
install(DIRECTORY include/
  DESTINATION include
)

# Install the message files
install(DIRECTORY msg/
  DESTINATION share/${PROJECT_NAME}/msg
)

# Install Python executables
install(PROGRAMS
  scripts/feature_extraction.py
  scripts/dead_reckoning_lidar.py
  scripts/slam_node.py

  src/slam_objects.py
  src/cfar_wrapper.py
  #src/slam_ros.py
  #src/slam.py
  #src/lidar_sonar_tools.py
  
  src/utils/conversions.py
  src/utils/topics.py
  src/utils/visualization.py

  scripts/object_slam_node.py
  src/object_slam.py
  src/scene_graph.py
  
  DESTINATION lib/${PROJECT_NAME})

target_link_libraries(cfar PUBLIC)
target_link_libraries(pcl PUBLIC ${libpointmatcher_LIBRARIES} ${PCL_LIBRARIES})
install(TARGETS
  cfar
  pcl
  # feature_extraction_node
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY launch DESTINATION share/${PROJECT_NAME}/)
install(DIRECTORY config DESTINATION share/${PROJECT_NAME}/)

ament_export_dependencies(rosidl_default_runtime)
ament_package()