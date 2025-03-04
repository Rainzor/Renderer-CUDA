cmake_minimum_required(VERSION 3.5)
project(Renderer)

# Enable C++17 for host code
set(CMAKE_CXX_STANDARD 17)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

#  ./install
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/install/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/install/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/install/bin)


# Set a default build type if none was specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    SET(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    # Set the possible values of build type for cmake-gui
    SET_PROPERTY(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

########################################
# CUDA Setup
########################################
find_package(CUDA 11 REQUIRED)
include(${CMAKE_MODULE_PATH}/CUDAComputesList.cmake)

list(APPEND CUDA_NVCC_FLAGS ${CUDA_GENERATE_CODE})
list(APPEND CUDA_NVCC_FLAGS_DEBUG "-g -G")
set(CUDA_VERBOSE_BUILD ON)

if(WIN32)
    # Set up include and lib paths
    set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER} CACHE FILEPATH "Host side compiler used by NVCC" FORCE)
endif(WIN32)
########################################

find_package(OpenGL REQUIRED)
# find_package(glfw3 REQUIRED)
# find_package(GLEW REQUIRED)
if(UNIX)
    find_package(glfw3 REQUIRED)
    find_package(GLEW REQUIRED)
    set(LIBRARIES glfw ${GLEW_LIBRARIES} ${OPENGL_gl_LIBRARY})
else(UNIX)
    set(EXTERNAL "external")

    set(GLFW_ROOT_DIR ${EXTERNAL})
    set(GLFW_USE_STATIC_LIBS ON)
    find_package(GLFW REQUIRED)

    set(GLEW_ROOT_DIR ${EXTERNAL})
    set(GLEW_USE_STATIC_LIBS ON)
    find_package(GLEW REQUIRED)

    add_definitions(${GLEW_DEFINITIONS})
    include_directories(${GLEW_INCLUDE_DIR} ${GLFW_INCLUDE_DIR})
    set(LIBRARIES ${GLEW_LIBRARY} ${GLFW_LIBRARY} ${OPENGL_LIBRARY})
endif(UNIX)

set(GLM_ROOT_DIR "external")
find_package(GLM REQUIRED)
include_directories(${GLM_INCLUDE_DIRS})

set(JSON_ROOT_DIR "external")
include_directories(${JSON_ROOT_DIR}/include/nlohmann)

include_directories(src)

file(GLOB headers
    src/*.h
    src/core/*.h
    src/core/*.cuh
    src/ImGui/*.h
    src/utils/*.h
    src/utils/*.hpp
)

file(GLOB sources
    src/*.cpp
    src/ImGui/*.cpp
    src/core/*.cu
    src/core/*.cpp
    src/utils/*.cpp
    )

list(SORT headers)
list(SORT sources)

source_group(Headers FILES ${headers})
source_group(Sources FILES ${sources})

add_subdirectory(external)
set(LIBRARIES ${LIBRARIES} ImGui)
#add_subdirectory(stream_compaction)  # TODO: uncomment if using your stream compaction

cuda_add_executable(${CMAKE_PROJECT_NAME} ${sources} ${headers})
target_link_libraries(${CMAKE_PROJECT_NAME}
    ${LIBRARIES}
    #stream_compaction  # TODO: uncomment if using your stream compaction
    )
