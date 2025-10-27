#ifndef SCHEDULE_HPP
#define SCHEDULE_HPP

#include "common.hpp"

#define N_THREADS_X 32
#define N_THREADS_Y 8

typedef struct {
    int n_blocks_x;
    int n_blocks_y;
    int n_threads_x;
    int n_threads_y;
} ScheduleDim;

void schedule_dim(uint width, uint height);

#endif  // SCHEDULE_HPP