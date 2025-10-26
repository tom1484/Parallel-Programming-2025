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

    int n_pixels = width * height;
    int n_threads = N_THREADS_X * N_THREADS_Y;
    
    int n_blocks_x = sqrt(n_blocks);
    int n_blocks_y = (n_blocks + n_blocks_x - 1) / n_blocks_x;

    int n_cluster = (n_pixels + n_threads - 1) / n_threads;
    // Square grid if possible
    int n_cluster_x = sqrt(n_cluster);
    int n_cluster_y = (n_cluster + n_cluster_x - 1) / n_cluster_x;
    
    n_blocks_x = std::min(n_blocks_x, n_cluster_x);
    n_blocks_y = std::min(n_blocks_y, n_cluster_y);

    dim.n_clusters_x = n_cluster_x;
    dim.n_clusters_y = n_cluster_y;
    dim.n_blocks_x = n_blocks_x;
    dim.n_blocks_y = n_blocks_y;
    dim.n_threads_x = N_THREADS_X;
    dim.n_threads_y = N_THREADS_Y;
    dim.batch_size_x = N_THREADS_X * n_blocks_x;
    dim.batch_size_y = N_THREADS_Y * n_blocks_y;
}