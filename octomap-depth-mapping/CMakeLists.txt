cmake_minimum_required(VERSION 3.18) # CUDA as language needs >= 3.18
project(octomap_depth_mapping LANGUAGES CXX CUDA)

# Default standards
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Enable CUDA build option
option(USE_CUDA "Enable CUDA acceleration" ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(message_filters REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)
find_package(octomap REQUIRED)
find_package(octomap_msgs REQUIRED)

include_directories(include)

if(USE_CUDA)
  add_compile_definitions(CUDA)

  # CUDA library (modern way)
  add_library(octomap_demap SHARED
    src/${PROJECT_NAME}.cpp
    src/cuda_proj.cu
  )

  set_target_properties(octomap_demap PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "61;75;86"
  )

else()
  add_library(octomap_demap SHARED
    src/${PROJECT_NAME}.cpp
  )
endif()

# --- CORRECTION ---
# To resolve the keyword signature conflict, we separate Eigen's dependency management.
# 1. Add Eigen's include directories to the target.
target_include_directories(octomap_demap PUBLIC
  ${Eigen3_INCLUDE_DIRS}
)
# 2. Link the Eigen library using the "plain" signature to match ament_target_dependencies.
target_link_libraries(octomap_demap
  Eigen3::Eigen
)
# --- END CORRECTION ---

ament_target_dependencies(octomap_demap
  rclcpp
  rclcpp_components
  message_filters
  cv_bridge
  octomap
  OpenCV
  octomap_msgs
  sensor_msgs
  geometry_msgs
  std_srvs
)

ament_export_targets(octomap_demap HAS_LIBRARY_TARGET)

rclcpp_components_register_node(octomap_demap
  PLUGIN "octomap_depth_mapping::OctomapDemap"
  EXECUTABLE ${PROJECT_NAME}
)

install(
  DIRECTORY include/
  DESTINATION include
)

install(
  TARGETS octomap_demap
  EXPORT octomap_demap
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include
)

install(TARGETS ${PROJECT_NAME}
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}/
)

ament_package()