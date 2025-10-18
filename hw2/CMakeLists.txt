cmake_minimum_required(VERSION 3.16)
set(CMAKE_CXX_COMPILER mpicxx)
project(hw2 CXX)

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

# Specify source files
set(SRC
  "${CMAKE_SOURCE_DIR}/src/hw2.cpp"
  "${CMAKE_SOURCE_DIR}/src/sift.cpp"
  "${CMAKE_SOURCE_DIR}/src/image.cpp"
)

# Define the executable target
add_executable(hw2 ${SRC})

# Include directory equivalent to -Iinclude
target_include_directories(hw2 PRIVATE "${CMAKE_SOURCE_DIR}/include")

# Define the validate executable (Release build only, excluded from default "all" target)
if(CMAKE_BUILD_TYPE STREQUAL "Release")
  add_executable(validate EXCLUDE_FROM_ALL "${CMAKE_SOURCE_DIR}/src/validate.cpp")
  target_include_directories(validate PRIVATE "${CMAKE_SOURCE_DIR}/include")
endif()

# MPI
if(APPLE)
  target_include_directories(hw2 PRIVATE /opt/homebrew/include)
endif()
find_package(MPI REQUIRED)
target_link_libraries(hw2 PRIVATE MPI::MPI_CXX)

# OpenMP
if(APPLE)
  target_include_directories(hw2 PRIVATE /opt/homebrew/opt/libomp/include)
  # For C and C++ with Clang on macOS, use “-Xpreprocessor -fopenmp” and link “omp”
  set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp")
  set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp")
  set(OpenMP_C_LIB_NAMES "omp")
  set(OpenMP_CXX_LIB_NAMES "omp")
  # Optional: if you know where libomp is installed:
  set(OpenMP_omp_LIBRARY /opt/homebrew/opt/libomp/lib/libomp.dylib)
endif()

find_package(OpenMP REQUIRED)
target_link_libraries(hw2 PRIVATE OpenMP::OpenMP_CXX)

# Optimize like -O3 when building Release
if(CMAKE_BUILD_TYPE STREQUAL "Release")
  target_compile_options(hw2
    PRIVATE $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-O3>)
endif()

# Turn on DEBUG flag when building Debug
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  target_compile_definitions(hw2 PRIVATE DEBUG)
  target_compile_options(hw2
    PRIVATE $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-g -Wall>)
endif()
