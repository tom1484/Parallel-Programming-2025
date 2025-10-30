#include <cstdint>
#include <cstdio>
#include <cstring>

#include "png_writer.hpp"

// Fast CRC32 table (precomputed)
static uint32_t crc_table[256];
static bool crc_table_computed = false;

static void make_crc_table() {
    if (crc_table_computed) return;

    for (uint32_t n = 0; n < 256; n++) {
        uint32_t c = n;
        for (int k = 0; k < 8; k++) {
            if (c & 1)
                c = 0xedb88320L ^ (c >> 1);
            else
                c = c >> 1;
        }
        crc_table[n] = c;
    }
    crc_table_computed = true;
}

static uint32_t update_crc(uint32_t crc, const unsigned char* buf, size_t len) {
    uint32_t c = crc;

    for (size_t n = 0; n < len; n++) {
        c = crc_table[(c ^ buf[n]) & 0xff] ^ (c >> 8);
    }
    return c;
}

static uint32_t crc(const unsigned char* buf, size_t len) { return update_crc(0xffffffffL, buf, len) ^ 0xffffffffL; }

static void write_be32(unsigned char* buf, uint32_t val) {
    buf[0] = (val >> 24) & 0xff;
    buf[1] = (val >> 16) & 0xff;
    buf[2] = (val >> 8) & 0xff;
    buf[3] = val & 0xff;
}

// Write a PNG chunk with CRC
static void write_chunk(FILE* fp, const char* type, const unsigned char* data, uint32_t length) {
    unsigned char buf[4];

    // Write length
    write_be32(buf, length);
    fwrite(buf, 1, 4, fp);

    // Write type
    fwrite(type, 1, 4, fp);

    // Calculate CRC (type + data)
    uint32_t crc_val = update_crc(0xffffffffL, (const unsigned char*)type, 4);
    if (length > 0) {
        crc_val = update_crc(crc_val, data, length);
    }
    crc_val ^= 0xffffffffL;

    // Write data
    if (length > 0) {
        fwrite(data, 1, length, fp);
    }

    // Write CRC
    write_be32(buf, crc_val);
    fwrite(buf, 1, 4, fp);
}

// Fast PNG writer without compression
void write_png_custom(const char* filename, unsigned char* raw_image, unsigned width, unsigned height) {
    make_crc_table();

    FILE* fp = fopen(filename, "wb");
    if (!fp) {
        fprintf(stderr, "Cannot open file: %s\n", filename);
        return;
    }

    // PNG signature
    const unsigned char png_sig[8] = {137, 80, 78, 71, 13, 10, 26, 10};
    fwrite(png_sig, 1, 8, fp);

    // IHDR chunk (image header)
    unsigned char ihdr[13];
    write_be32(ihdr + 0, width);   // Width
    write_be32(ihdr + 4, height);  // Height
    ihdr[8] = 8;                   // Bit depth
    ihdr[9] = 6;                   // Color type (RGBA)
    ihdr[10] = 0;                  // Compression method
    ihdr[11] = 0;                  // Filter method
    ihdr[12] = 0;                  // Interlace method
    write_chunk(fp, "IHDR", ihdr, 13);

    // IDAT chunk (image data) - uncompressed deflate stream
    // Each scanline: 1 filter byte + width * 4 bytes (RGBA)
    uint32_t scanline_size = 1 + width * 4;
    uint32_t raw_size = scanline_size * height;

    // Uncompressed deflate blocks have a 65535 byte limit per block
    // We need to split data into multiple blocks if it exceeds this
    const uint32_t MAX_BLOCK_SIZE = 65535;
    uint32_t num_blocks = (raw_size + MAX_BLOCK_SIZE - 1) / MAX_BLOCK_SIZE;
    
    // Calculate total zlib size:
    // - 2 bytes: zlib header
    // - For each block: 5 bytes (header + len + nlen) + data
    // - 4 bytes: Adler-32 checksum
    uint32_t zlib_size = 2 + (num_blocks * 5) + raw_size + 4;

    // Allocate buffer for entire zlib stream
    unsigned char* zlib_buf = new unsigned char[zlib_size];
    unsigned char* p = zlib_buf;

    // Zlib header (RFC 1950)
    *p++ = 0x78;  // CMF: CM=8 (deflate), CINFO=7 (32K window)
    *p++ = 0x01;  // FLG: FLEVEL=0, FDICT=0, FCHECK=1

    // Build the complete image data first in a temp buffer for Adler-32
    unsigned char* temp_data = new unsigned char[raw_size];
    unsigned char* temp_ptr = temp_data;
    
    for (uint32_t y = 0; y < height; y++) {
        *temp_ptr++ = 0;  // Filter type 0 (None)
        memcpy(temp_ptr, raw_image + y * width * 4, width * 4);
        temp_ptr += width * 4;
    }
    
    // Now write deflate blocks with data interleaved
    uint32_t remaining = raw_size;
    uint32_t offset = 0;
    
    for (uint32_t block_idx = 0; block_idx < num_blocks; block_idx++) {
        uint32_t block_size = (remaining > MAX_BLOCK_SIZE) ? MAX_BLOCK_SIZE : remaining;
        bool is_final = (block_idx == num_blocks - 1);
        
        // Block header: BFINAL (1 bit) + BTYPE (2 bits, 00 = no compression)
        *p++ = is_final ? 0x01 : 0x00;
        
        // LEN (little-endian, 16-bit)
        *p++ = block_size & 0xff;
        *p++ = (block_size >> 8) & 0xff;
        
        // NLEN (one's complement of LEN, little-endian)
        uint16_t nlen = ~block_size;
        *p++ = nlen & 0xff;
        *p++ = (nlen >> 8) & 0xff;
        
        // Copy this block's data
        memcpy(p, temp_data + offset, block_size);
        p += block_size;
        
        remaining -= block_size;
        offset += block_size;
    }

    // Calculate Adler-32 checksum (RFC 1950)
    // Adler-32 is calculated over the uncompressed data (after deflate decompression)
    // In our case, that's the raw image data with filter bytes
    uint32_t s1 = 1;
    uint32_t s2 = 0;
    const uint32_t BASE = 65521;  // largest prime smaller than 65536
    
    // Process data in chunks to avoid overflow
    // We can safely do about 5500 iterations before s2 might overflow
    const uint32_t NMAX = 5552;
    uint32_t n = raw_size;
    const unsigned char* adler_ptr = temp_data;
    
    while (n >= NMAX) {
        for (uint32_t i = 0; i < NMAX; i++) {
            s1 += *adler_ptr++;
            s2 += s1;
        }
        s1 %= BASE;
        s2 %= BASE;
        n -= NMAX;
    }
    
    // Process remaining bytes
    for (uint32_t i = 0; i < n; i++) {
        s1 += *adler_ptr++;
        s2 += s1;
    }
    s1 %= BASE;
    s2 %= BASE;
    
    uint32_t adler = (s2 << 16) | s1;
    write_be32(p, adler);  // Adler-32 is big-endian
    p += 4;
    
    // Clean up
    delete[] temp_data;

    // Write IDAT chunk with the zlib stream
    write_chunk(fp, "IDAT", zlib_buf, zlib_size);
    delete[] zlib_buf;

    // IEND chunk (end of file)
    write_chunk(fp, "IEND", nullptr, 0);

    fclose(fp);
}
