cmake_minimum_required(VERSION 3.16)

project(hw3 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
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
  "${CMAKE_SOURCE_DIR}/lodepng/lodepng.cpp"
  "${CMAKE_SOURCE_DIR}/src/global.cpp"
  "${CMAKE_SOURCE_DIR}/src/io.cpp"
  "${CMAKE_SOURCE_DIR}/src/hw3.cu"
  "${CMAKE_SOURCE_DIR}/src/utils.cu"
  "${CMAKE_SOURCE_DIR}/src/schedule.cu"
  "${CMAKE_SOURCE_DIR}/src/render.cu"
)
set(CPU_SRC
  "${CMAKE_SOURCE_DIR}/lodepng/lodepng.cpp"
  "${CMAKE_SOURCE_DIR}/src/hw3_cpu.cpp"
)

message(${CPU_SRC})

# Define the executable target
add_executable(hw3 ${SRC})
add_executable(hw3_cpu EXCLUDE_FROM_ALL ${CPU_SRC})

# Include directory equivalent to -Iinclude
target_include_directories(hw3 PRIVATE "${CMAKE_SOURCE_DIR}/include")
target_include_directories(hw3 PRIVATE "${CMAKE_SOURCE_DIR}/glm/include")
target_include_directories(hw3 PRIVATE "${CMAKE_SOURCE_DIR}/lodepng")

target_include_directories(hw3_cpu PRIVATE "${CMAKE_SOURCE_DIR}/include")
target_include_directories(hw3_cpu PRIVATE "${CMAKE_SOURCE_DIR}/glm/include")
target_include_directories(hw3_cpu PRIVATE "${CMAKE_SOURCE_DIR}/lodepng")

# If user passed -DZLIB (e.g. -DZLIB=ON), locate ZLIB and add the ZLIB compile definition
if(DEFINED ZLIB AND ZLIB)
  find_package(ZLIB REQUIRED)
  target_compile_definitions(hw3 PRIVATE ZLIB)
  target_link_libraries(hw3 PRIVATE ZLIB::ZLIB)
endif()

# Release mode adds -O3 optimization by default
if(CMAKE_BUILD_TYPE STREQUAL "Release")
  target_compile_options(hw3
    PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:
        -w
        -lineinfo
        --extra-device-vectorization
        -use_fast_math
        -fmad true
        --generate-line-info
        # -Xptxas --register-usage-level=10,--warn-on-spills,--optimize-float-atomics,-v
        -Xptxas -O3,-maxrregcount=48,--optimize-float-atomics,-v
        >
  )
endif()

# Turn on DEBUG flag when building Debug
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  target_compile_definitions(hw3 PRIVATE DEBUG)
  target_compile_options(hw3
    PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:-g -Wall>
      $<$<COMPILE_LANGUAGE:CUDA>:-g -w -use_fast_math -Xptxas -v>
  )
endif()
