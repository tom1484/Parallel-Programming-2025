#include <iostream>

#include "schedule.hpp"

extern ScheduleDim dim;

void schedule_dim(uint width, uint height) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    int n_blocks = prop.multiProcessorCount;
    int max_threads = prop.maxThreadsPerBlock;
    int warp_size = prop.warpSize;

    assert((N_THREADS_X * N_THREADS_Y) <= max_threads);

    int n_blocks_x = (width + N_THREADS_X - 1) / N_THREADS_X;
    int n_blocks_y = (height + N_THREADS_Y - 1) / N_THREADS_Y;

    dim.n_blocks_x = n_blocks_x;
    dim.n_blocks_y = n_blocks_y;
    dim.n_threads_x = N_THREADS_X;
    dim.n_threads_y = N_THREADS_Y;
}