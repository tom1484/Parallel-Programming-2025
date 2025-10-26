#ifndef SCHEDULE_HPP
#define SCHEDULE_HPP

#include "common.hpp"

#define N_THREADS_X 16
#define N_THREADS_Y 16

/**
 * Schedule dimensions for CUDA kernel execution.
 *
 * Members:
 * - n_clusters_x: number of clusters in x dimension
 * - n_clusters_y: number of clusters in y dimension
 * - n_blocks_x: number of blocks in x dimension
 * - n_blocks_y: number of blocks in y dimension
 * - n_threads_x: number of threads per block in x dimension
 * - n_threads_y: number of threads per block in y dimension
 * - batch_size_x: size of one batch in x dimension (n_threads_x * n_blocks_x)
 * - batch_size_y: size of one batch in y dimension (n_threads_y * n_blocks_y)
 *
 * Usage:
 * - pixel_x = batch_size_x * batch_x + n_threads_x * blockIdx.x + threadIdx.x
 * - pixel_y = batch_size_y * batch_y + n_threads_y * blockIdx.y + threadIdx.y
 */
typedef struct {
    int n_clusters_x;
    int n_clusters_y;
    int n_blocks_x;
    int n_blocks_y;
    int n_threads_x;
    int n_threads_y;
    int batch_size_x;
    int batch_size_y;
} ScheduleDim;

void schedule_dim(uint width, uint height);

#endif  // SCHEDULE_HPP