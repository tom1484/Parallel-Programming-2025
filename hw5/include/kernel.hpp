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

// Optionally: if you use CUDA-style <<< >>> syntax in your code,
// you can define a no-op that will help IntelliSense parse it.
// But depending on your build setup, omitting this may be safer:
// #define kernelName<<<...>>>(...)   /* no-op */

// If you want, define CUDA-style keywords to something harmless
#define __global__
#define __device__
#define __shared__
#define __constant__
#define __host__

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

__device__ double gravity_device_mass_dev(double m0, double t);

__global__ void nbody_step_kernel(int n, int step, double* qx, double* qy, double* qz, double* vx, double* vy,
                                  double* vz, const double* m, const int* type, bool ignore_devices,
                                  int disabled_device);

void run_step_gpu(int step, int n, DeviceArrays& dev, bool ignore_devices, int disabled_device);

#endif  // KERNEL_HPP