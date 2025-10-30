#include <cstdio>

#include "common.hpp"
#include "utils.hpp"

void ProgressBar::update(int current) {
    float percent = (current + initial) * 100.0f / total;
    std::cerr << "\rProgress: " << std::setw(6) << std::setprecision(2) << std::fixed << percent << "%";
    if (oneline) {
        std::cerr << std::flush;
    } else {
        std::cerr << std::endl;
    }
}

void ProgressBar::done() { std::cerr << std::endl; }

void write_png_fast(const char* filename, unsigned char* raw_image, unsigned width, unsigned height) {
    LodePNGState state;
    lodepng_state_init(&state);

    // No compression - fastest encoding
    state.encoder.zlibsettings.btype = 0;      // Disable compression
    state.encoder.filter_strategy = LFS_ZERO;  // No filtering
    state.encoder.auto_convert = 0;            // Skip color analysis

    // Set color mode explicitly
    state.info_raw.colortype = LCT_RGBA;
    state.info_raw.bitdepth = 8;
    state.info_png.color.colortype = LCT_RGBA;
    state.info_png.color.bitdepth = 8;

    unsigned char* png_buffer;
    size_t png_size;
    unsigned error = lodepng_encode(&png_buffer, &png_size, raw_image, width, height, &state);

    if (!error) {
        error = lodepng_save_file(png_buffer, png_size, filename);
    }

    // lodepng_free(png_buffer);
    // lodepng_state_cleanup(&state);

    if (error) {
        printf("png error %u: %s\n", error, lodepng_error_text(error));
    }
}

// Save raw_image to PNG file
void write_png(const char* filename, unsigned char* raw_image, unsigned width, unsigned height) {
    unsigned error = lodepng_encode32_file(filename, raw_image, width, height);
    if (error) printf("png error %u: %s\n", error, lodepng_error_text(error));
}

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

void estimate_occupancy(void* kernel, int block_size, int dynamic_smem) {
    int activeBlocksPerSM = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&activeBlocksPerSM, kernel,
                                                  block_size,   // e.g. 128 threads/block
                                                  dynamic_smem  // bytes of dynamic shared memory per block
    );

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int maxThreadsPerSM = prop.maxThreadsPerMultiProcessor;

    float occupancy = (activeBlocksPerSM * block_size) / (float)maxThreadsPerSM;
    printf("Theoretical occupancy: %.2f %%\n", occupancy * 100.0f);
}