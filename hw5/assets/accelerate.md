# N-Body Simulator - GPU Acceleration

## Quick Summary

This n-body simulator uses **HIP/ROCm** to accelerate gravitational simulation on AMD GPUs. The key optimizations are:

1. **Mega-kernels**: Entire 200k-step simulation runs in a single GPU kernel launch (no CPU sync per step)
2. **Shared memory**: Body positions/masses cached in fast on-chip memory
3. **Multi-GPU**: Problem 3 splits device testing across 2 GPUs using OpenMP threads
4. **HIP streams**: 4 concurrent streams per GPU overlap multiple simulations

## Problem Description

The simulator solves three related problems:

1. **Problem 1**: Find minimum distance between planet and asteroid over 200,000 time steps (ignoring gravity devices)
2. **Problem 2**: Detect if/when asteroid collides with planet (with gravity devices active)
3. **Problem 3**: Find the cheapest gravity device to destroy with a missile to prevent collision

## Code Structure

```
src/
├── kernel.cpp      # GPU kernels and multi-GPU orchestration
├── main.cpp        # Host code, memory setup, problem dispatch
└── utils.cpp       # File I/O

include/
└── kernel.hpp      # Kernel declarations, DeviceArrays struct
```

### Key Functions in `kernel.cpp`

| Function | Purpose |
|----------|---------|
| `simulate_problem1_kernel` | Mega-kernel: find min distance (ignores devices) |
| `simulate_problem2_kernel` | Mega-kernel: detect collision step (with devices) |
| `simulate_problem3_kernel` | Mega-kernel: test one device with missile logic |
| `run_problem3_multi_gpu` | Orchestrates 2 GPUs × 4 streams for Problem 3 |

### Data Flow

```
Problem 1 & 2:
  Host → GPU (once) → Mega-kernel (200k steps) → Result → Host

Problem 3:
  Host ──┬──→ GPU0 (streams 0-3) ──┬──→ Merge results
         └──→ GPU1 (streams 0-3) ──┘
```

## Optimization Journey

### Stage 1: Naive (Baseline)
```
CPU                          GPU
 |--launch step kernel------->|
 |<--copy ALL positions------|   ← 6n × 8 bytes per step!
 |  compute distance on CPU   |
 |  (repeat 200k times)       |
```

### Stage 2: Mega-Kernel
```
CPU                          GPU
 |--launch mega-kernel------->|   ← 1 launch for entire simulation!
 |                            |   [runs 200k steps internally]
 |<--copy final result-------|   ← 1 copy at the end
```

### Stage 3: Multi-GPU + Streams
```
CPU ──┬──→ GPU0 (4 streams) ──┬──→ Merge
      └──→ GPU1 (4 streams) ──┘
```

## Key Design Features

| Feature | Implementation |
|---------|---------------|
| **Shared Memory** | Body data (qx, qy, qz, m, type) loaded into `__shared__` arrays |
| **Synchronization** | `__syncthreads()` between physics update and condition check |
| **Early Exit** | Problems 2 & 3 break loop when collision detected |
| **Multi-GPU** | OpenMP threads, each controlling one GPU |
| **HIP Streams** | 4 streams per GPU for concurrent device simulations |

## Performance Results

| Testcase | Optimization | Time | Speedup |
|----------|--------------|------|---------|
| b20 | Baseline | 30.3s | 1.0x |
| b20 | Final (all opts) | **5.6s** | **5.4x** |
| b80 | Baseline | ~120s | 1.0x |
| b80 | Final (all opts) | **21.1s** | **5.7x** |

## Limitations

1. **Single Block Constraint**: Limited to n ≤ 1024 bodies (max threads per block)
2. **Problem 3 Re-runs**: Each device requires a full simulation (no state caching)

## API Reference

### Problem 1: Minimum Distance
```cpp
double run_simulation_problem1(
    int n_steps, int n, int planet, int asteroid,
    DeviceArrays& dev, double* d_result
);
```

### Problem 2: Collision Detection
```cpp
int run_simulation_problem2(
    int n_steps, int n, int planet, int asteroid,
    double planet_radius, DeviceArrays& dev, int* d_result
);
```

### Problem 3: Multi-GPU Missile Defense
```cpp
void run_problem3_multi_gpu(
    int n_steps, int n, int planet, int asteroid,
    const int* device_ids, int num_devices,
    double planet_radius, double missile_speed, double dt,
    const double* h_qx, const double* h_qy, const double* h_qz,
    const double* h_vx, const double* h_vy, const double* h_vz,
    const double* h_m, const int* h_type,
    int* out_best_device_idx, int* out_best_hit_step
);
```
