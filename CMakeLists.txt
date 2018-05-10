project(csva)
include("${PROJECT_SOURCE_DIR}/cmake/macro.cmake")
include_directories(${PROJECT_SOURCE_DIR}/modules)

if (CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++14 -g -fPIC -fext-numeric-literals" CACHE STRING COMPILE_FLAGS FORCE)
    set(CMAKE_C_FLAGS "${CMAKE_CXX_FLAGS} -Wall -g -fPIC" CACHE STRING COMPILE_FLAGS FORCE)

    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -g -march=native --fast-math -ffast-math -mtune=native -funroll-loops -DNDEBUG -DBOOST_DISABLE_ASSERTS" CACHE STRING COMPILE_FLAGS FORCE)
    set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -DDEBUG" CACHE STRING COMPILE_FLAGS FORCE)

elseif (MSVC)

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Ox /Ob2 /Oi /Ot /arch:AVX /fp:fast /EHsc /W4" CACHE STRING COMPILE_FLAGS FORCE)

    ADD_DEFINITIONS(
        -D_USE_MATH_DEFINES
        -DNOMINMAX
        -DLANG_CXX11
    )
endif()

cmake_minimum_required(VERSION 3.4)
set(CMAKE_CXX_STANDARD 11)
message("${PROJECT_SOURCE_DIR}")
if (EXISTS "${PROJECT_SOURCE_DIR}/cmake/config/config.cmake")
	include("${PROJECT_SOURCE_DIR}/cmake/config/config.cmake")
endif()
if(WIN32)
  list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake/config")
elseif(UNIX)
  list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake/config_lin")
endif()

project (csva CXX)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
add_subdirectory(modules)
add_subdirectory(examples)
