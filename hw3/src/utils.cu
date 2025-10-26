#include <cstdio>

#include "utils.hpp"

void print_device_info() {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaGetDeviceCount error: %s\n", cudaGetErrorString(err));
        deviceCount = 0;
    }

    fprintf(stderr, "CUDA device count: %d\n\n", deviceCount);
    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, dev);

        fprintf(stderr, "Device %d: %s\n", dev, prop.name);
        fprintf(stderr, "  Compute capability: %d.%d\n", prop.major, prop.minor);
        fprintf(stderr, "  Total global memory: %zu bytes\n", prop.totalGlobalMem);
        fprintf(stderr, "  Shared memory per block: %zu bytes\n", prop.sharedMemPerBlock);
        fprintf(stderr, "  Registers per block: %d\n", prop.regsPerBlock);
        fprintf(stderr, "  Warp size: %d\n", prop.warpSize);
        fprintf(stderr, "  Memory pitch: %zu\n", prop.memPitch);
        fprintf(stderr, "  Max threads per block: %d\n", prop.maxThreadsPerBlock);
        fprintf(stderr, "  Max threads dim (x,y,z): %d %d %d\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1],
                prop.maxThreadsDim[2]);
        fprintf(stderr, "  Max grid size (x,y,z): %d %d %d\n", prop.maxGridSize[0], prop.maxGridSize[1],
                prop.maxGridSize[2]);
        fprintf(stderr, "  Multiprocessor count: %d\n", prop.multiProcessorCount);
        fprintf(stderr, "  Max threads per multiprocessor: %d\n", prop.maxThreadsPerMultiProcessor);
        fprintf(stderr, "  Concurrent kernels: %s\n", prop.concurrentKernels ? "Yes" : "No");
        fprintf(stderr, "  PCI bus id: %d, PCI device id: %d\n", prop.pciBusID, prop.pciDeviceID);

        // choose a default thread count based on device capability
        unsigned int num_threads = static_cast<unsigned int>(prop.maxThreadsPerBlock);
        fprintf(stderr, "  Selected num_threads = %u\n\n", num_threads);
    }
}