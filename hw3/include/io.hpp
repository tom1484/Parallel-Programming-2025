#ifndef IO_HPP
#define IO_HPP

#include <cstring>

#ifdef ZLIB
#include <zlib.h>
#endif

void write_png(const char* filename, unsigned char* raw_image, unsigned width, unsigned height);
void write_png_fast(const char* filename, unsigned char* raw_image, unsigned width, unsigned height);
void write_png_custom(const char* filename, unsigned char* raw_image, unsigned width, unsigned height);

#endif