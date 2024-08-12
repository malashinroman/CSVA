PROJECT(match_aero)
cmake_minimum_required(VERSION 3.4)

FIND_PACKAGE(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

if (UNIX)
  find_package(Boost REQUIRED COMPONENTS filesystem system)
	INCLUDE_DIRECTORIES( ${Boost_INCLUDE_DIRS} )
endif (UNIX)

if (WIN32)
	find_package(Boost REQUIRED)
	include_directories(${Boost_INCLUDE_DIRS})
	set (CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS OFF)
	link_directories(${Boost_LIBRARY_DIR})
	# message(${Boost_LIBRARY_DIR})
endif (WIN32)

# if (UNIX)
#    add_compile_options(-fext-numeric-literals)
# endif (UNIX)

add_compile_options(-fopenmp)
#if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
#else ()
# add_compile_options(-openmp)
#endif()

add_compile_options(-std=c++11)
ADD_EXECUTABLE(match_aero aerospace_demo.cpp)
TARGET_LINK_LIBRARIES(match_aero csva_lib common_lib ${OpenCV_LIBS} ${Boost_LIBRARIES} stdc++fs)
