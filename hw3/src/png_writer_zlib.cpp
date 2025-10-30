#include <zlib.h>  // make sure you link with -lz (zlib) or equivalent

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "png_writer.hpp"

static uint32_t crc32_table[256];
static bool crc32_table_computed = false;

static void make_crc32_table() {
    for (uint32_t n = 0; n < 256; n++) {
        uint32_t c = n;
        for (int k = 0; k < 8; k++) {
            if (c & 1)
                c = 0xEDB88320UL ^ (c >> 1);
            else
                c = c >> 1;
        }
        crc32_table[n] = c;
    }
    crc32_table_computed = true;
}

static uint32_t update_crc(uint32_t crc, const unsigned char* buf, size_t len) {
    uint32_t c = crc ^ 0xFFFFFFFFUL;
    if (!crc32_table_computed) make_crc32_table();
    for (size_t n = 0; n < len; n++) {
        c = crc32_table[(c ^ buf[n]) & 0xFF] ^ (c >> 8);
    }
    return c ^ 0xFFFFFFFFUL;
}

static void write_uint32_be(FILE* f, uint32_t v) {
    unsigned char b[4];
    b[0] = (v >> 24) & 0xFF;
    b[1] = (v >> 16) & 0xFF;
    b[2] = (v >> 8) & 0xFF;
    b[3] = v & 0xFF;
    fwrite(b, 1, 4, f);
}

void write_png_custom(const char* filename, unsigned char* raw_image, unsigned width, unsigned height) {
    FILE* f = std::fopen(filename, "wb");
    if (!f) {
        std::fprintf(stderr, "Failed to open file %s for writing\n", filename);
        return;
    }

    // 1. PNG signature
    const unsigned char png_sig[8] = {0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A};
    fwrite(png_sig, 1, 8, f);

    // 2. IHDR chunk
    uint32_t ihdr_data_len = 13;
    write_uint32_be(f, ihdr_data_len);
    const char ihdr_type[4] = {'I', 'H', 'D', 'R'};
    fwrite(ihdr_type, 1, 4, f);
    unsigned char ihdr_data[13];
    // width (4 bytes)
    ihdr_data[0] = (width >> 24) & 0xFF;
    ihdr_data[1] = (width >> 16) & 0xFF;
    ihdr_data[2] = (width >> 8) & 0xFF;
    ihdr_data[3] = (width) & 0xFF;
    // height
    ihdr_data[4] = (height >> 24) & 0xFF;
    ihdr_data[5] = (height >> 16) & 0xFF;
    ihdr_data[6] = (height >> 8) & 0xFF;
    ihdr_data[7] = (height) & 0xFF;
    ihdr_data[8] = 8;   // bit-depth
    ihdr_data[9] = 6;   // color-type = RGBA
    ihdr_data[10] = 0;  // compression method
    ihdr_data[11] = 0;  // filter method
    ihdr_data[12] = 0;  // interlace method
    fwrite(ihdr_data, 1, 13, f);
    uint32_t ihdr_crc = update_crc(0, (const unsigned char*)ihdr_type, 4);
    ihdr_crc = update_crc(ihdr_crc, ihdr_data, 13);
    write_uint32_be(f, ihdr_crc);

    // 3. IDAT chunk: prepare image data with filter bytes + compress with zlib level 0
    // Each scanline: one leading filter byte (0), then width*4 bytes of pixel data
    size_t scanline_len = 1 + width * 4;
    size_t raw_data_size = scanline_len * height;
    std::vector<unsigned char> raw_buf;
    raw_buf.resize(raw_data_size);
    unsigned char* p = raw_buf.data();
    for (unsigned y = 0; y < height; y++) {
        *p++ = 0;  // filter type 0
        unsigned char* row_ptr = raw_image + (size_t)y * width * 4;
        std::memcpy(p, row_ptr, width * 4);
        p += width * 4;
    }

    // compress
    uLongf comp_bound = compressBound(raw_data_size);
    std::vector<unsigned char> comp_buf;
    comp_buf.resize(comp_bound);
    // use zlib with level 0 = no compression (fastest) (Z_DEFAULT_STRATEGY typically)
    int ret = compress2(comp_buf.data(), &comp_bound, raw_buf.data(), raw_data_size, 0);
    if (ret != Z_OK) {
        std::fprintf(stderr, "zlib compress2 failed: %d\n", ret);
        std::fclose(f);
        return;
    }
    comp_buf.resize(comp_bound);

    // write chunk header
    write_uint32_be(f, (uint32_t)comp_buf.size());
    const char idat_type[4] = {'I', 'D', 'A', 'T'};
    fwrite(idat_type, 1, 4, f);
    fwrite(comp_buf.data(), 1, comp_buf.size(), f);
    uint32_t idat_crc = update_crc(0, (const unsigned char*)idat_type, 4);
    idat_crc = update_crc(idat_crc, comp_buf.data(), comp_buf.size());
    write_uint32_be(f, idat_crc);

    // 4. IEND chunk
    write_uint32_be(f, 0);
    const char iend_type[4] = {'I', 'E', 'N', 'D'};
    fwrite(iend_type, 1, 4, f);
    uint32_t iend_crc = update_crc(0, (const unsigned char*)iend_type, 4);
    write_uint32_be(f, iend_crc);

    std::fclose(f);
}