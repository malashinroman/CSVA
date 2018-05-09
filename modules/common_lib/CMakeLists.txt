PROJECT( common_lib )
message("----------------common_lib------------")
cmake_minimum_required (VERSION 2.6)

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

ADD_LIBRARY(common_lib 
			DIRENT.H DirectoryParser.h DirectoryParser.cpp 
			feature_extractors.h feature_extractors.cpp 
			misc.h misc.cpp)

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

TARGET_LINK_LIBRARIES(common_lib  ${Boost_LIBRARIES} ${TinyXML_LIBRARIES} ${OpenCV_LIBS})