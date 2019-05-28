cmake_minimum_required(VERSION 3.0.0)
project(camera_model)

# Build setting
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

# set dictionary 
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

# C++ 11
add_definitions(-std=c++11)

# Debug and log messages
if(CMAKE_BUILD_TYPE MATCHES "Debug")
	add_definitions(-D_DEBUG_MODE_)
endif(CMAKE_BUILD_TYPE MATCHES "Debug")

# cmake modules
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# OpenCV Library
find_package(OpenCV 3.2.0 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Eigen Library
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

# glog
find_package(Glog 0.3.5 REQUIRED)

# include
include_directories(
    ${PROJECT_SOURCE_DIR}/include
)

# add librarys
set(LINK_LIBS ${OpenCV_LIBS} ${GLOG_LIBRARY})
set(CAMMODEL_CPP src/abstract_camera.cpp src/MEI_camera.cpp src/pinhole_camera.cpp)

add_library(${PROJECT_NAME} STATIC ${CAMMODEL_CPP})
target_link_libraries(${PROJECT_NAME} ${LINK_LIBS}) 

# build test
add_executable(test test/remove_distort.cpp)
target_link_libraries(test ${PROJECT_NAME})