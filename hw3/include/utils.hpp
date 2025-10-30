#ifndef UTILS_HPP
#define UTILS_HPP

#include <iomanip>
#include <iostream>

#define CUDA_CHECK(err)                                                                 \
    if (err != cudaSuccess) {                                                                 \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE);                                                                   \
    }

// Simple progress bar that only show percentage
class ProgressBar {
   private:
    int total;
    int initial;
    bool oneline;

   public:
    ProgressBar(int total, int initial = 0) : total(total), initial(initial), oneline(true) {}
    ProgressBar(int total, bool oneline, int initial = 0) : total(total), initial(initial), oneline(oneline) {}
    void update(int current);
    void done();
};

void print_device_info();
void estimate_occupancy(void* kernel, int block_size, int dynamic_smem);

#endif