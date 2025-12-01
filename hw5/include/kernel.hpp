#ifndef KERNEL_HPP
#define KERNEL_HPP

// Intellisense hack for dim3 and built-in variables
#ifndef __HIPCC__  // or some macro that hipcc defines
// Provide dummy definitions for Intellisense
struct dim3 {
    unsigned int x, y, z;
};
extern dim3 blockIdx, threadIdx, blockDim, gridDim;
#endif

#include <hip/hip_runtime.h>

__device__ double gravity_device_mass_dev(double m0, double t);

__global__ void nbody_step_kernel(int n, int step, double* qx, double* qy, double* qz, double* vx, double* vy,
                                  double* vz, const double* m, const int* type, bool ignore_devices,
                                  int disabled_device);

#endif  // KERNEL_HPP