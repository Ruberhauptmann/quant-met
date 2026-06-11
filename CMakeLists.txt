cmake_minimum_required(VERSION 3.27)
project(quant_met_core LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 1. Find dependencies
find_package(pybind11 CONFIG REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)
if(APPLE)
    # Point CMake explicitly to Homebrew's OpenMP directory location
    # so find_package handles architecture probing safely under the hood.
    if(EXISTS "/opt/homebrew/opt/libomp")
        set(OpenMP_ROOT "/opt/homebrew/opt/libomp")
    else()
        set(OpenMP_ROOT "/usr/local/opt/libomp")
    endif()
endif()
find_package(OpenMP REQUIRED)

# 2. Define the pybind11 module
pybind11_add_module(quant_met_core src/bindings.cpp src/main.cpp)

# Direct the output binary straight into the python package directory
set_target_properties(quant_met_core PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/quant_met"
)

target_include_directories(quant_met_core PRIVATE include)

target_link_libraries(quant_met_core PRIVATE
        OpenMP::OpenMP_CXX
        Eigen3::Eigen
)
