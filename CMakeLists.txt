cmake_minimum_required(VERSION 3.0.0)
project(camera_model)

# Build setting
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

# C++ 11
add_definitions(-std=c++11)

# Debug and log messages
if(CMAKE_BUILD_TYPE MATCHES "Debug")
	add_definitions(-D_DEBUG_MODE_)
endif(CMAKE_BUILD_TYPE MATCHES "Debug")

# OpenCV Library
find_package(OpenCV 3.2.0 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Eigen Library
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

# glog
find_package(Glog 0.3.5 REQUIRED)

include_directories(
    ${PROJECT_SOURCE_DIR}/include
)

add_executable()
target_link_libraries()