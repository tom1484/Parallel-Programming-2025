// Your cuda program :)

#ifndef SUBMIT
#include <chrono>
#include <iostream>
#endif

#include "common.hpp"
#include "render.hpp"
#include "schedule.hpp"
#include "utils.hpp"

using namespace std;

extern uint width;        // image width
extern uint height;       // image height
extern vec2 iResolution;  // just for convenience of calculation

extern vec3 camera_pos;  // camera position in 3D space (x, y, z)
extern vec3 target_pos;  // target position in 3D space (x, y, z)

extern ScheduleDim dim;

uchar* raw_image;  // 1D image
uchar** image;     // 2D image

int main(int argc, char** argv) {
#ifndef SUBMIT
    auto __start = chrono::high_resolution_clock::now();
#endif

    assert(argc == 10);

    // Init parameters
    camera_pos = vec3(atof(argv[1]), atof(argv[2]), atof(argv[3]));
    target_pos = vec3(atof(argv[4]), atof(argv[5]), atof(argv[6]));
    width = atoi(argv[7]);
    height = atoi(argv[8]);

    float total_pixel = width * height;
    float current_pixel = 0;

    iResolution = vec2(width, height);

    // Create image
    raw_image = new uchar[width * height * 4];
    image = new uchar*[height];

    for (int i = 0; i < height; ++i) {
        image[i] = raw_image + i * width * 4;
    }

    schedule_dim(width, height);
#ifdef DEBUG
    cout << "Grid: (" << dim.n_blocks_x << ", " << dim.n_blocks_y << "), Block: (" << dim.n_threads_x << ", "
         << dim.n_threads_y << ")" << endl;
#endif
    copy_constants_to_device();

    for (int bx = 0; bx < width; bx += dim.batch_size_x) {
        for (int by = 0; by < height; by += dim.batch_size_y) {
            uchar* buffer[dim.batch_size_y];
            for (int j = 0; j < dim.batch_size_y && (by + j) < height; j++) {
                buffer[j] = &image[by + j][(bx) * 4];
            }
            int valid_size_x = min(dim.batch_size_x, (int)(width - bx));
            int valid_size_y = min(dim.batch_size_y, (int)(height - by));
            render_batch(buffer, valid_size_x, valid_size_y);
        }
    }

    // Save image and finalize
    write_png(argv[9], raw_image, width, height);
    delete[] raw_image;
    delete[] image;

#ifndef SUBMIT
    auto __end = chrono::high_resolution_clock::now();
    auto __elapsed_us = chrono::duration_cast<chrono::microseconds>(__end - __start);
    cerr << "Elapsed: " << __elapsed_us.count() << " us" << endl;
#endif

    return 0;
}