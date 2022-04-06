cmake_minimum_required(VERSION 3.8.0)
project(lidar_clustering)

find_package(autoware_build_flags REQUIRED)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  grid_map_cv
  grid_map_msgs
  grid_map_ros
  itolab_senior_car_msgs
  jsk_rviz_plugins
  pcl_ros
  roscpp
  sensor_msgs
  std_msgs
  tf
  vector_map_server
)

find_package(OpenMP)
find_package(OpenCV REQUIRED)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "-O3 -Wall ${CMAKE_CXX_FLAGS}")
set(CMAKE_EXPORT_COMPILE_COMMANDS "ON")

catkin_package(
 INCLUDE_DIRS include
 CATKIN_DEPENDS
   geometry_msgs
   grid_map_cv
   grid_map_msgs
   grid_map_ros
   itolab_senior_car_msgs
   jsk_rviz_plugins
   pcl_ros
   roscpp
   sensor_msgs
   std_msgs
   tf
   vector_map_server
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

link_directories(${OpenCV_LIBRARY_DIRS})

add_executable(${PROJECT_NAME} src/${PROJECT_NAME}.cpp
  src/${PROJECT_NAME}_node.cpp
  src/${PROJECT_NAME}_node_main.cpp)

find_package(CUDA)
find_package(Eigen3 QUIET)

AW_CHECK_CUDA()

if(USE_CUDA)
  message("-- USING ACCELERATED CLUSTERING --")
  message("Version: " ${CUDA_VERSION})
  message("Library: " ${CUDA_CUDA_LIBRARY})
  message("Runtime: " ${CUDA_CUDART_LIBRARY})
  target_compile_definitions(${PROJECT_NAME} PRIVATE GPU_CLUSTERING=0
    )

  cuda_add_library(gpu_${PROJECT_NAME}
    include/${PROJECT_NAME}/gpu_${PROJECT_NAME}.h
    src/gpu_${PROJECT_NAME}.cu
    )

  target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBRARIES}
    ${catkin_LIBRARIES}
    gpu_${PROJECT_NAME}
    )

  install(TARGETS
    gpu_${PROJECT_NAME}
    ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
    LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
    RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
    )
else()
  target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBRARIES}
    ${catkin_LIBRARIES}
    )
endif()

install(DIRECTORY include/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  )

install(TARGETS
  ${PROJECT_NAME}
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
  )

install(DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
  PATTERN ".svn" EXCLUDE
  )
