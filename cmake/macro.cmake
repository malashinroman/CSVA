# === set_max_warnings_level ===
macro(set_maximum_warnings_level)
    if (MSVC)
        if (CMAKE_CXX_FLAGS MATCHES "/W[0-4]")
            string(REGEX REPLACE "/W[0-4]" "/W4" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
        else ()
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")
        endif ()
    elseif (CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wconversion -Wno-long-long -pedantic")
    endif ()
endmacro()

# === set_msvc_definitions ===
macro(set_msvc_specific_defines)
    if (MSVC)
        add_definitions(-DNOMINMAX)
        add_definitions(-D_SCL_SECURE_NO_WARNINGS)
		add_definitions(-D_CRT_SECURE_NO_WARNINGS)
    endif ()
endmacro()

# === configure_msvc_runtime ===
macro(configure_msvc_runtime MSVC_RUNTIME)
    if (MSVC)
        # Default to statically-linked runtime.
        if ("${MSVC_RUNTIME}" STREQUAL "")
            set(MSVC_RUNTIME "static")
        endif ()
        # Set compiler options.
        set(variables
                CMAKE_C_FLAGS_DEBUG
                CMAKE_C_FLAGS_MINSIZEREL
                CMAKE_C_FLAGS_RELEASE
                CMAKE_C_FLAGS_RELWITHDEBINFO
                CMAKE_CXX_FLAGS_DEBUG
                CMAKE_CXX_FLAGS_MINSIZEREL
                CMAKE_CXX_FLAGS_RELEASE
                CMAKE_CXX_FLAGS_RELWITHDEBINFO
                )

        if (${MSVC_RUNTIME} STREQUAL "static")
            message(STATUS
                    "MSVC -> forcing use of statically-linked runtime."
                    )
            foreach (variable ${variables})
                if (${variable} MATCHES "/MD")
                    string(REGEX REPLACE "/MD" "/MT" ${variable} "${${variable}}")
                endif ()
            endforeach ()
        else ()
            message(STATUS
                    "MSVC -> forcing use of dynamically-linked runtime."
                    )
            foreach (variable ${variables})
                if (${variable} MATCHES "/MT")
                    string(REGEX REPLACE "/MT" "/MD" ${variable} "${${variable}}")
                endif ()
            endforeach ()
        endif ()
    endif ()
endmacro()

# === register_test_target ===
macro(register_test_target NAME LIBRARIES_TO_LINK)
    set(TARGET_NAME test_${NAME})

    add_executable(${TARGET_NAME} ${TARGET_NAME}.cpp)

    target_link_libraries(${TARGET_NAME} gtest gmock gtest_main gmock_main)

    foreach (LIB ${LIBRARIES_TO_LINK})
        target_link_libraries(${TARGET_NAME} "${LIB}")
    endforeach ()

    add_test(NAME ${TARGET_NAME} COMMAND $<TARGET_FILE:${TARGET_NAME}>)
    set_property(TARGET ${TARGET_NAME} PROPERTY FOLDER "tests")
endmacro()

# === link opencv ===
macro(link_opencv)
    set(OpenCV_STATIC ON)
    find_package(OpenCV REQUIRED)
    include_directories(${OpenCV_INCLUDE_DIRS})
    link_directories(${OpenCV_LIB_DIR})
endmacro()

# === link_openmp ===
macro(link_openmp)
    find_package(OpenMP)
    if (OPENMP_FOUND)
        message(STATUS "OpenMP is ON")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    endif ()
endmacro()

# === link_opengl ===
macro(link_opengl)
    find_package(OpenGL)
    if (${OPENGL_FOUND})
        message(STATUS "OpenGL is ON")
        include_directories(${OpenGL_INCLUDE_DIRS})
        link_directories(${OpenGL_LIB_DIR})
        add_subdirectory(${PROJECT_SOURCE_DIR}/externals/glfw)
        include_directories(${PROJECT_SOURCE_DIR}/externals/glfw/include)   
        include_directories(${PROJECT_SOURCE_DIR}/externals/gl)
    endif ()
endmacro()

# === link_opencl ===
macro(link_opencl)
    find_package(OpenCL)
    if (${OpenCL_FOUND})
        message(STATUS "OpenCL is ON")
        include_directories(${OpenCL_INCLUDE_DIR})
        link_directories(${OpenCL_LIB_DIR})
    endif ()
endmacro()
	