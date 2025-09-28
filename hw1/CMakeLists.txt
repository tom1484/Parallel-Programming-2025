cmake_minimum_required(VERSION 3.16)
project(hw1 CXX)

# Build type (default to Release if not set)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE
      Release
      CACHE STRING "Build type" FORCE)
endif()

# C++ standard = C++17 (users can still override via -DCMAKE_CXX_STANDARD=...)
set(CMAKE_CXX_STANDARD 17) # or target property CXX_STANDARD
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Output compile_commands for LSP indexing
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Collect sources like your Makefile's wildcard
file(GLOB SRC CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/src/*.cpp")

# Define the executable target
add_executable(hw1 ${SRC}) # creates target "hw1"

# Include directory equivalent to -Iinclude
target_include_directories(hw1 PRIVATE "${CMAKE_SOURCE_DIR}/include")

# Threads / pthreads
set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
set(THREADS_PREFER_PTHREAD_FLAG TRUE)
find_package(Threads REQUIRED)
target_link_libraries(hw1 PRIVATE Threads::Threads)

# OpenMP
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  target_link_libraries(hw1 PRIVATE OpenMP::OpenMP_CXX)
endif()

# Optimize like -O3 when building Release
if(CMAKE_BUILD_TYPE STREQUAL "Release")
  target_compile_options(hw1
                         PRIVATE $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-O3>)
endif()

# Turn on DEBUG flag when building Debug
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  target_compile_definitions(hw1 PRIVATE DEBUG)
  target_compile_options(hw1
                         PRIVATE $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-g -Wall -Wextra -Wpedantic>)
endif()
