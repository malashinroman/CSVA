PROJECT(match_aero)
cmake_minimum_required(VERSION 3.4)

FIND_PACKAGE(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(Boost REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIR})

ADD_EXECUTABLE(match_aero aerospace_demo.cpp)
TARGET_LINK_LIBRARIES(match_aero csva_lib common_lib ${OpenCV_LIBS} ${Boost_LIBRARIES})
