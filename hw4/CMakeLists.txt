cmake_minimum_required(VERSION 3.16)

project(hw4 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_ARCHITECTURES 70)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Build type (default to Release if not set)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE
    Release
    CACHE STRING "Build type" FORCE)
endif()

# Output compile_commands for LSP indexing
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Specify source files for CUDA
set(SRC
  "${CMAKE_SOURCE_DIR}/src/main.cu"
)
set(SAMPLE_SRC
  "${CMAKE_SOURCE_DIR}/src/sample/main.cu"
  "${CMAKE_SOURCE_DIR}/src/sample/sha256.cu"
)

# Define the executable target
add_executable(hw4 ${SRC})
add_executable(sample EXCLUDE_FROM_ALL ${SAMPLE_SRC})

# Include directory equivalent to -Iinclude
target_include_directories(hw4 PRIVATE "${CMAKE_SOURCE_DIR}/include")
target_include_directories(sample PRIVATE "${CMAKE_SOURCE_DIR}/include/sample")

target_compile_options(hw4
  PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
      -w
      --extra-device-vectorization
      -use_fast_math
      -fmad true
      --generate-line-info
      -Xptxas -O3,-maxrregcount=48,--optimize-float-atomics,-v
      >
)
target_compile_options(sample
  PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-w -Xptxas -O3,-v>
)
