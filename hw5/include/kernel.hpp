#ifndef KERNEL_HPP
#define KERNEL_HPP

// Intellisense hack for dim3 and built-in variables
// If an actual HIP compiler is used, skip this file entirely
#if !defined(__HIPCC__) && !defined(__CUDACC__)

// Dummy definitions to pacify IntelliSense / LSP
namespace hip {
struct dim3 {
    unsigned x, y, z;
    dim3(unsigned _x = 1, unsigned _y = 1, unsigned _z = 1) : x(_x), y(_y), z(_z) {}
};
using hipStream_t = void*;
}  // namespace hip

// Provide dummy built-in variables / structs often used in kernels
extern const hip::dim3 blockIdx, threadIdx, blockDim, gridDim;

// Provide a dummy macro for hipLaunchKernelGGL so editor doesn't error
#define hipLaunchKernelGGL(kernelName, gridDim, blockDim, dynamicShared, stream, ...) /* no-op; for IntelliSense only \
                                                                                       */

#define __longlong_as_double(x) (static_cast<double>(x))
#define __double_as_longlong(x) (static_cast<unsigned long long>(x))

#define atomicCAS(address, compare, val) (*(address) == (compare) ? (*(address) = (val), (compare)) : *(address))

// If you want, define CUDA-style keywords to something harmless
#define __global__
#define __device__
#define __shared__
#define __constant__
#define __host__
#define __syncthreads()

#endif

#include <hip/hip_runtime.h>

// Accept the output of expression (e.g., a HIP runtime call) and check for errors
#ifdef SUBMIT
#define CHECK(expr) (expr)
#else

#include <iostream>
#define CHECK(expr)                                                                               \
    do {                                                                                          \
        hipError_t err = (expr);                                                                  \
        if (err != hipSuccess) {                                                                  \
            fprintf(stderr, "HIP error %s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(err)); \
            exit(EXIT_FAILURE);                                                                   \
        }                                                                                         \
    } while (0)

#endif

struct DeviceArrays {
    double *qx, *qy, *qz;
    double *vx, *vy, *vz;
    double* m;
    int* type;
};

// Problem 3 result structure
struct Problem3Result {
    bool saved;           // true if planet survived
    int missile_hit_step; // step when missile hit device (-1 if never)
    int collision_step;   // step when asteroid hit planet (-1 if never)
};

// Mega-kernel functions - run entire simulation in a single kernel launch
double run_simulation_problem1(int n_steps, int n, int planet, int asteroid,
                               DeviceArrays& dev, double* d_result);

int run_simulation_problem2(int n_steps, int n, int planet, int asteroid,
                            double planet_radius, DeviceArrays& dev, int* d_result);

// Allocate device arrays on current GPU
void allocate_device_arrays_gpu(DeviceArrays& dev, int n);

// Free device arrays
void free_device_arrays(DeviceArrays& dev);

// Copy host data to device arrays
void copy_host_to_device_arrays(DeviceArrays& dev, int n,
                                const double* qx, const double* qy, const double* qz,
                                const double* vx, const double* vy, const double* vz,
                                const double* m, const int* type);

// Multi-GPU Problem 3: Split work across multiple GPUs with HIP streams
void run_problem3_multi_gpu(int n_steps, int n, int planet, int asteroid,
                            const int* device_ids, int num_devices,
                            double planet_radius, double missile_speed, double dt,
                            const double* h_qx, const double* h_qy, const double* h_qz,
                            const double* h_vx, const double* h_vy, const double* h_vz,
                            const double* h_m, const int* h_type,
                            int* out_best_device_idx, int* out_best_hit_step);

#endif  // KERNEL_HPP
