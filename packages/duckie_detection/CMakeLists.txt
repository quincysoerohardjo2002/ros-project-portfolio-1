cmake_minimum_required(VERSION 2.8.3)
project(duckie_detection)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
  duckietown_msgs
)

catkin_python_setup()

catkin_package()

install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)
