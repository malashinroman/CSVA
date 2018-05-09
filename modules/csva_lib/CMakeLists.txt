PROJECT( csva_lib )
message("----------------csva_lib------------")
add_definitions(-DBUILD_WITH_CERES)
cmake_minimum_required (VERSION 2.6)
if(WIN32)
  list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
elseif(UNIX)
  list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake_lin")
endif()
FIND_PACKAGE(OpenCV 3.1 REQUIRED core imgproc highgui features2d xfeatures2d ml video shape stitching calib3d)

if (UNIX)
	find_package(Boost REQUIRED COMPONENTS filesystem)
	INCLUDE_DIRECTORIES( ${Boost_INCLUDE_DIRS} )
	#find_package(Eigen REQUIRED)
endif (UNIX)

if (WIN32)
	find_package(Boost REQUIRED)
	include_directories(${Boost_INCLUDE_DIRS})
	set (CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS OFF)
	link_directories(${Boost_LIBRARY_DIR})
	message(${Boost_LIBRARY_DIR})
endif (WIN32)

add_compile_options(-std=c++11)
find_package(TinyXML REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${TinyXML_INCLUDE_DIRS})
include_directories(${grayscale_pca_lib_INCLUDE_DIRS})

if(BUILD_WITH_CERES)
	find_package(Ceres REQUIRED)
	if(Ceres_FOUND)
		message(${CERES_LIBRARIES})
		message(Ceres found)
	else()
		message(Ceres not found)
	endif()
	
	include_directories(${CERES_INCLUDE_DIRS})
	
	find_package (glog 0.3.4 REQUIRED)
	find_package(GFLAGS REQUIRED)
	include_directories(${GFLAGS_INCLUDE_DIR})
	
	message(${GFLAGS_LIBRARIES})
	
	find_package(Eigen3 REQUIRED)
	include_directories(${EIGEN3_INCLUDE_DIR})
	message(${GFLAGS_LIBRARIES})
	message(${GFLAGS_INCLUDE_DIR})
endif()

ADD_LIBRARY(csva_lib SHARED csva.h csva.cpp Cluster_data.cpp Cluster_data.h HoughList.cpp HoughList.h  
matching_hough.cpp 
matching_hough.h misc_functions.cpp misc_functions.h misc_visualizers.h
misc_visualizers.cpp
confidence_estimation.h
confidence_estimation.cpp
)
#message(${SOURCE})
#message("Boost_FOUND")

if(OpenCV_FOUND)
  message(Opencv found)
endif()
if(TinyXML_FOUND)
  message(TinyXML_FOUND)
  #message(${TinyXML_INCLUDE_DIRS})
else()
  message("tinyxml_not_found")
endif()
if(BoostFOUND)
  message(Boost found)
else()
  message(Boost not found)
endif()

TARGET_LINK_LIBRARIES(csva_lib  ${Boost_LIBRARIES} ${TinyXML_LIBRARIES} ${OpenCV_LIBS} ${CERES_LIBRARIES} glog::glog ${GFLAGS_LIBRARIES} ${CERES_LIBRARIES} grayscale_pca_lib)
set(${PROJECT_NAME}_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/include
    				CACHE INTERNAL "${PROJECT_NAME}: Include Directories" FORCE)