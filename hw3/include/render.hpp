#ifndef RENDER_HPP
#define RENDER_HPP

#include "common.hpp"

void copy_constants_to_device();
void render_batch(uchar* buffer[], int size_x, int size_y);

#endif  // RENDER_HPP