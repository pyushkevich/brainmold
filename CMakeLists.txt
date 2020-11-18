# Requisite CMake commands
cmake_minimum_required(VERSION 3.16)

# Project name
project(brainmold)

# Where to get additional modules
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/CMake/")

# Find ITK and include needed stuff
find_package(ITK REQUIRED)
include(${ITK_USE_FILE})

# Find C3D and include needed stuff
find_package(Convert3D REQUIRED)
include_directories(${Convert3D_INCLUDE_DIRS})

# Find IPOPT
find_package(IpOpt REQUIRED)
include_directories(${IPOPT_INCLUDE_DIR})

# Add the executable
add_executable(brainmold brainmold.cxx GentleNLP.cxx IPOptProblemInterface.cxx)
target_link_libraries(brainmold ${ITK_LIBRARIES} ${Convert3D_LIBRARIES} ${IPOPT_LIBRARIES})



