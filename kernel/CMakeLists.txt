#cmake -DCMAKE_BUILD_TYPE=Debug ../<source>
#cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo ../<source>
#cmake -DCMAKE_BUILD_TYPE=Release ../<source>

cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
project(Kernel)

set(CMAKE_CXX_STANDARD 14)  # enable C++11 standard
set(CMAKE_CXX_STANDARD_REQUIRED ON)
#set(CMAKE_CXX_EXTENSIONS ON)


# Set a default build type if none was specified
set(default_build_type "RelWithDebInfo")
   
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
   message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
   set_property(CACHE CMAKE_BUILD_TYPE PROPERTY VALUE "${default_build_type}")
endif()



set(COMPILE_FLAGS_COMMON "-Wno-unused-result -fPIC -fopenmp")

set(COMPILE_FLAGS_GONE32 "${COMPILE_FLAGS_COMMON} -DB32")
set(COMPILE_FLAGS_GONE64 "${COMPILE_FLAGS_COMMON} -DB64")

set(LINK_FLAGS_COMMON "-fopenmp")
link_directories(${CMAKE_SOURCE_DIR})

set(kernel_source_files
    kernel_pybind.cpp
)

add_subdirectory(pybind11)

pybind11_add_module(kernel ${kernel_source_files})
set_target_properties(kernel PROPERTIES  COMPILE_FLAGS ${COMPILE_FLAGS_GONE32}
                      LINK_FLAGS ${LINK_FLAGS_COMMON})
target_link_libraries(kernel PRIVATE spmm)

