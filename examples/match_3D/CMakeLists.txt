PROJECT( match3D )
cmake_minimum_required(VERSION 3.4)

FIND_PACKAGE( OpenCV REQUIRED core imgproc highgui)
include_directories(${OpenCV_INCLUDE_DIRS})

add_compile_options(-std=c++11)
ADD_EXECUTABLE(match3D main.cpp 3Drecognition.h 3Drecognition.cpp)
TARGET_LINK_LIBRARIES(match3D csva_lib common_lib ${OpenCV_LIBS})