# N-Body Simulator - GPU Acceleration Implementation

## Overview

This document explains the GPU acceleration strategy used in the n-body simulator. The implementation evolved through several optimization stages, culminating in a **mega-kernel** approach that runs entire simulations in a single GPU kernel launch.

## Problem Description

The simulator solves three related problems:

1. **Problem 1**: Find minimum distance between planet and asteroid over 200,000 time steps (ignoring gravity devices)
2. **Problem 2**: Detect if/when asteroid collides with planet (with gravity devices active)
3. **Problem 3**: Find the cheapest gravity device to destroy with a missile to prevent collision

## Optimization Journey

### Stage 1: Naive Implementation (Baseline)
```
CPU                          GPU
 |                            |
 |--launch step kernel------->|
 |<--copy ALL positions------|   ← 6n × 8 bytes per step!
 |  compute distance on CPU   |
 |  (repeat 200k times)       |
```
**Problem**: Copying all body positions back to CPU every step is extremely slow.

### Stage 2: GPU Distance Computation
```
CPU                          GPU
 |                            |
 |--launch step kernel------->|
 |--launch distance kernel--->|
 |<--copy 8 bytes------------|   ← Only 1 double!
 |  (repeat 200k times)       |
```
**Improvement**: ~6000x less data transfer per step.

### Stage 3: Batched Simulation
```
CPU                          GPU
 |                            |
 |--launch step kernel------->|
 |--launch check kernel------>|
 |  (repeat internally)       |
 |<--copy result (periodic)---|   ← Check every 1000 steps
```
**Improvement**: Reduced CPU-GPU round trips by 1000x.

### Stage 4: Mega-Kernel (Current Implementation)
```
CPU                          GPU
 |                            |
 |--launch mega-kernel------->|   ← 1 launch for entire simulation!
 |                            |   [runs 200k steps internally]
 |<--copy final result-------|   ← 1 copy at the end
```
**Improvement**: Eliminates ALL intermediate synchronization.

## Current Architecture

### Mega-Kernel Design

Each problem has its own mega-kernel that runs the entire simulation:

```cpp
__global__ void simulate_problem1_kernel(...) {
    __shared__ double s_min_dist;  // Shared state
    
    for (int step = 0; step <= n_steps; step++) {
        // 1. Each thread updates one body
        if (i < n) {
            // Compute acceleration, update velocity & position
        }
        __syncthreads();  // Barrier: all bodies updated
        
        // 2. Thread 0 checks condition
        if (i == 0) {
            // Update min_dist or check collision
        }
        __syncthreads();  // Barrier: condition checked
    }
    
    // Write final result
    if (i == 0) *result = s_min_dist;
}
```

### Key Features

| Feature | Implementation |
|---------|---------------|
| **Synchronization** | `__syncthreads()` between physics update and condition check |
| **Shared State** | `__shared__` variables for min_dist, collision_step, missile_hit |
| **Early Exit** | Problems 2 & 3 break loop when collision detected |
| **Single Block** | All threads in one block (max 1024 bodies) |

### File Structure

```
src/
├── kernel.cpp      # GPU kernels (mega-kernels for all 3 problems)
├── main.cpp        # Host code, memory management, problem orchestration
└── utils.cpp       # File I/O

include/
└── kernel.hpp      # Kernel declarations, DeviceArrays struct
```

## Performance Results

Tested on testcase b20:

| Implementation | Time | Speedup |
|----------------|------|---------|
| Naive (copy every step) | 30.3s | 1.0x |
| Batched (periodic checks) | 10.9s | 2.8x |
| **Mega-kernel** | **7.1s** | **4.3x** |

## Limitations

1. **Single Block Constraint**: Current implementation limited to n ≤ 1024 bodies (max threads per block)
2. **No Shared Memory Tiling**: Inner loop still has O(n²) global memory accesses
3. **Problem 3 Overhead**: Each device requires a full simulation re-run

## Future Optimizations

1. **Cooperative Groups**: Use `hipLaunchCooperativeKernel` for grid-wide sync (n > 1024)
2. **Shared Memory Tiling**: Load body positions into shared memory in tiles
3. **Problem 3 Parallelization**: Run multiple device simulations concurrently
4. **State Checkpointing**: Cache simulation states to avoid full re-runs

## API Reference

### Problem 1: Minimum Distance
```cpp
double run_simulation_problem1(
    int n_steps,        // Number of simulation steps
    int n,              // Number of bodies
    int planet,         // Planet body index
    int asteroid,       // Asteroid body index
    DeviceArrays& dev,  // GPU arrays
    double* d_result    // Device pointer for result
);
// Returns: Minimum distance between planet and asteroid
```

### Problem 2: Collision Detection
```cpp
int run_simulation_problem2(
    int n_steps,
    int n,
    int planet,
    int asteroid,
    double planet_radius,  // Collision threshold
    DeviceArrays& dev,
    int* d_result
);
// Returns: Step of collision, or -2 if no collision
```

### Problem 3: Missile Defense
```cpp
Problem3Result run_simulation_problem3(
    int n_steps,
    int n,
    int planet,
    int asteroid,
    int device_id,         // Device to destroy
    double planet_radius,
    double missile_speed,
    double dt,
    DeviceArrays& dev,
    int* d_result
);
// Returns: Problem3Result { saved, missile_hit_step, collision_step }
```
