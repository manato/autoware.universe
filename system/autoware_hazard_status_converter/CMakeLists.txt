cmake_minimum_required(VERSION 3.14)
project(autoware_hazard_status_converter)

find_package(autoware_cmake REQUIRED)
autoware_package()

ament_auto_add_library(${PROJECT_NAME} SHARED
  src/converter.cpp
)

rclcpp_components_register_node(${PROJECT_NAME}
  PLUGIN "autoware::hazard_status_converter::Converter"
  EXECUTABLE ${PROJECT_NAME}_node
)

ament_auto_package(INSTALL_TO_SHARE launch)
