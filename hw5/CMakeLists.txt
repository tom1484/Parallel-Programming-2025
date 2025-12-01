cmake_minimum_required(VERSION 3.16)

project(hw5 LANGUAGES CXX)

# --------------------------------------------------------------------
# Compilers
# --------------------------------------------------------------------
# Use hipcc as the global C++ compiler (for hw5)
# Adjust the path if your hipcc is elsewhere.
set(CMAKE_CXX_COMPILER "/usr/bin/hipcc" CACHE STRING "C++ compiler" FORCE)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Build type (default to Release if not set)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE
      Release
      CACHE STRING "Build type" FORCE)
endif()

# Output compile_commands for LSP indexing
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# --------------------------------------------------------------------
# hw5: built with hipcc
# --------------------------------------------------------------------
set(SRC
    "${CMAKE_SOURCE_DIR}/src/main.cpp"
)

add_executable(hw5 ${SRC})

# Include directory equivalent to -Iinclude
target_include_directories(hw5 PRIVATE "${CMAKE_SOURCE_DIR}/include")
# Add HIP include directories manually (for LSP + hipcc builds)
target_include_directories(hw5 PRIVATE /opt/rocm/include)

# Tell hipcc which GPU arch to target
target_compile_options(hw5 PRIVATE --offload-arch=gfx908)

# (Optional) extra HIP-related flags can go here, for example:
# target_compile_options(hw5 PRIVATE -O3)

# --------------------------------------------------------------------
# sample: built with g++ (manual custom target)
# --------------------------------------------------------------------
set(SAMPLE_SRC
    "${CMAKE_SOURCE_DIR}/src/sample/main.cc"
)

# We don't use CMake's CXX rules here because those use hipcc.
# Instead, we define a custom target that runs g++ directly.
add_custom_target(sample
    COMMAND g++ -std=c++17
            -I"${CMAKE_SOURCE_DIR}/include"
            -o build/sample
            "${SAMPLE_SRC}"
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    COMMENT "Building sample with g++"
)

# If you want 'sample' NOT to build by default, leave it as-is.
# To build it explicitly:
#   cmake --build . --target sample