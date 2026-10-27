cmake_minimum_required(VERSION 3.8)
project(perception_sim)

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY urdf worlds maps launch models
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
