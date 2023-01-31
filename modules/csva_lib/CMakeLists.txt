PROJECT( csva_lib )
cmake_minimum_required (VERSION 3.4)
if(WIN32)
  list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
elseif(UNIX)
  list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake_lin")
endif()
FIND_PACKAGE(OpenCV REQUIRED core imgproc highgui features2d xfeatures2d ml video shape calib3d)

if (UNIX)
	find_package(Boost REQUIRED COMPONENTS filesystem)
	INCLUDE_DIRECTORIES( ${Boost_INCLUDE_DIRS} )
endif (UNIX)

if (WIN32)
	find_package(Boost REQUIRED)
	include_directories(${Boost_INCLUDE_DIRS})
	set (CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS OFF)
	link_directories(${Boost_LIBRARY_DIR})
	# message(${Boost_LIBRARY_DIR})
endif (WIN32)

if (UNIX)
   add_compile_options(-fext-numeric-literals)
   add_compile_options(-fopenmp)
   find_package(OpenMP)
   #message(${OpenMP_CXX_LIBRARIES})
endif (UNIX)


include_directories(${OpenCV_INCLUDE_DIRS})

ADD_LIBRARY(csva_lib SHARED csva.h csva.cpp Cluster_data.cpp Cluster_data.h HoughList.cpp HoughList.h
                            matching_hough.cpp
                            matching_hough.h misc_functions.cpp misc_functions.h
                            confidence_estimation.h
                            confidence_estimation.cpp)

TARGET_LINK_LIBRARIES(csva_lib ${Boost_LIBRARIES} ${OpenCV_LIBS})
#set(${PROJECT_NAME}_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/include
#   				`CACHE INTERNAL "${PROJECT_NAME}: Include Directories" FORCE)
