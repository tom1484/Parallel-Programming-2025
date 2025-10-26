// Your cuda program :)

#ifndef SUBMIT
#include <chrono>
#include <iostream>
#endif

#include "common.h"

using namespace std;

unsigned int num_threads;  // number of thread
unsigned int width;        // image width
unsigned int height;       // image height
vec2 iResolution;          // just for convenience of calculation

vec3 camera_pos;  // camera position in 3D space (x, y, z)
vec3 target_pos;  // target position in 3D space (x, y, z)

unsigned char* raw_image;  // 1D image
unsigned char** image;     // 2D image

int main(int argc, char** argv) {
#ifndef SUBMIT
    auto start = chrono::high_resolution_clock::now();
#endif

    assert(argc == 10);

    // Init parameters
    camera_pos = vec3(atof(argv[1]), atof(argv[2]), atof(argv[3]));
    target_pos = vec3(atof(argv[4]), atof(argv[5]), atof(argv[6]));
    width = atoi(argv[7]);
    height = atoi(argv[8]);

    double total_pixel = width * height;
    double current_pixel = 0;

    iResolution = vec2(width, height);

    // Create image
    raw_image = new unsigned char[width * height * 4];
    image = new unsigned char*[height];

    for (int i = 0; i < height; ++i) {
        image[i] = raw_image + i * width * 4;
    }

#ifndef SUBMIT
    auto end = chrono::high_resolution_clock::now();
    auto elapsed_us = chrono::duration_cast<chrono::microseconds>(end - start);
    cerr << "Elapsed: " << elapsed_us.count() << " us" << endl;
#endif

    return 0;
}