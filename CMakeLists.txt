cmake_minimum_required(VERSION 3.27)
project(quant_met_core LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 1. Find dependencies
find_package(pybind11 CONFIG REQUIRED)
find_package(Eigen3 CONFIG REQUIRED) # Assumes Eigen is installed on your system

# --- Robust OpenMP Handling for macOS & Cross-Platform ---
if(APPLE)
    # Check for Homebrew's libomp path natively
    if(EXISTS "/opt/homebrew/opt/libomp")
        set(HOMEBREW_OMP_DIR "/opt/homebrew/opt/libomp")
    else()
        # Fallback for older Intel Macs using Homebrew
        set(HOMEBREW_OMP_DIR "/usr/local/opt/libomp")
    endif()

    if(EXISTS "${HOMEBREW_OMP_DIR}")
        message(STATUS "Found macOS Homebrew LibOpenMP at: ${HOMEBREW_OMP_DIR}")

        # Create a mock OpenMP target manually to bypass the broken find_package logic
        if(NOT TARGET OpenMP::OpenMP_CXX)
            add_library(OpenMP::OpenMP_CXX INTERFACE IMPORTED)
            set_target_properties(OpenMP::OpenMP_CXX PROPERTIES
                    INTERFACE_COMPILE_OPTIONS "-Xpreprocessor;-fopenmp"
                    INTERFACE_INCLUDE_DIRECTORIES "${HOMEBREW_OMP_DIR}/include"
                    INTERFACE_LINK_LIBRARIES "${HOMEBREW_OMP_DIR}/lib/libomp.dylib"
            )
        endif()
        set(OpenMP_CXX_FOUND TRUE)
    endif()
endif()

# Fallback to standard tracking for Linux/Windows if we didn't mock it on Mac
if(NOT OpenMP_CXX_FOUND)
    find_package(OpenMP REQUIRED)
endif()

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
