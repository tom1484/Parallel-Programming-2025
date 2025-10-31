#include "io.hpp"

#include "common.hpp"

// Save raw_image to PNG file
void write_png(const char* filename, unsigned char* raw_image, unsigned width, unsigned height) {
    unsigned error = lodepng_encode32_file(filename, raw_image, width, height);
    if (error) printf("png error %u: %s\n", error, lodepng_error_text(error));
}

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

#ifndef ZLIB

// Fast CRC32 table (precomputed)
static uint32_t crc_table[256];
static bool crc_table_computed = false;

static void make_crc_table() {
    if (crc_table_computed) return;
    for (uint32_t n = 0; n < 256; n++) {
        uint32_t c = n;
        for (int k = 0; k < 8; k++) {
            c = (c & 1) ? (0xedb88320u ^ (c >> 1)) : (c >> 1);
        }
        crc_table[n] = c;
    }
    crc_table_computed = true;
}

static inline uint32_t update_crc(uint32_t crc, const unsigned char* buf, size_t len) {
    uint32_t c = crc;
    for (size_t n = 0; n < len; n++) {
        c = crc_table[(c ^ buf[n]) & 0xffu] ^ (c >> 8);
    }
    return c;
}

static inline uint32_t crc(const unsigned char* buf, size_t len) {
    return update_crc(0xffffffffu, buf, len) ^ 0xffffffffu;
}

static inline void write_be32(unsigned char* buf, uint32_t val) {
    buf[0] = (unsigned char)((val >> 24) & 0xff);
    buf[1] = (unsigned char)((val >> 16) & 0xff);
    buf[2] = (unsigned char)((val >> 8) & 0xff);
    buf[3] = (unsigned char)(val & 0xff);
}

// Write a PNG chunk with CRC
static void write_chunk(FILE* fp, const char* type, const unsigned char* data, uint32_t length) {
    unsigned char hdr[4];

    // write length
    write_be32(hdr, length);
    fwrite(hdr, 1, 4, fp);

    // write type
    fwrite(type, 1, 4, fp);

    // CRC over type + data
    uint32_t c = update_crc(0xffffffffu, (const unsigned char*)type, 4);
    if (length) {
        c = update_crc(c, data, length);
    }
    c ^= 0xffffffffu;

    // data
    if (length) {
        fwrite(data, 1, length, fp);
    }

    // crc
    write_be32(hdr, c);
    fwrite(hdr, 1, 4, fp);
}

// Helper that emits an uncompressed DEFLATE block header at *p
// returns pointer after header
static inline unsigned char* emit_uncompressed_block_header(unsigned char* p, uint32_t block_size, bool is_final) {
    // 1 byte: BFINAL + BTYPE(00)
    *p++ = is_final ? 0x01 : 0x00;
    // LEN (LE)
    *p++ = (unsigned char)(block_size & 0xff);
    *p++ = (unsigned char)((block_size >> 8) & 0xff);
    // NLEN (LE, one's complement)
    uint16_t nlen = (uint16_t)~block_size;
    *p++ = (unsigned char)(nlen & 0xff);
    *p++ = (unsigned char)((nlen >> 8) & 0xff);
    return p;
}

// Fast PNG writer without compression
void write_png_custom(const char* filename, unsigned char* raw_image, unsigned width, unsigned height) {
    make_crc_table();

    FILE* fp = fopen(filename, "wb");
    if (!fp) {
        std::fprintf(stderr, "Cannot open file: %s\n", filename);
        return;
    }

    // PNG signature
    static const unsigned char png_sig[8] = {137, 80, 78, 71, 13, 10, 26, 10};
    fwrite(png_sig, 1, 8, fp);

    // IHDR
    unsigned char ihdr[13];
    write_be32(ihdr + 0, width);
    write_be32(ihdr + 4, height);
    ihdr[8] = 8;   // bit depth
    ihdr[9] = 6;   // RGBA
    ihdr[10] = 0;  // compression
    ihdr[11] = 0;  // filter
    ihdr[12] = 0;  // interlace
    write_chunk(fp, "IHDR", ihdr, 13);

    // image data sizes
    const uint32_t scanline_size = 1 + width * 4;  // 1 filter byte + RGBA
    const uint32_t raw_size = scanline_size * height;

    // DEFLATE uncompressed block limit
    const uint32_t MAX_BLOCK_SIZE = 65535u;
    const uint32_t num_blocks = (raw_size + MAX_BLOCK_SIZE - 1u) / MAX_BLOCK_SIZE;

    // zlib total size = 2(header) + num_blocks*5(block headers) + raw_size + 4(adler)
    const uint32_t zlib_size = 2u + num_blocks * 5u + raw_size + 4u;

    // allocate zlib buffer
    unsigned char* zlib_buf = new unsigned char[zlib_size];
    unsigned char* p = zlib_buf;

    // zlib header
    *p++ = 0x78;  // CMF
    *p++ = 0x01;  // FLG (store-only, FCHECK already ok)

    // Adler-32 state
    uint32_t s1 = 1;
    uint32_t s2 = 0;
    const uint32_t BASE = 65521u;
    const uint32_t NMAX = 5552u;

    // setup first block
    uint32_t remaining_bytes = raw_size;
    uint32_t current_block_remaining = (remaining_bytes > MAX_BLOCK_SIZE) ? MAX_BLOCK_SIZE : remaining_bytes;
    bool is_final_block = (remaining_bytes <= MAX_BLOCK_SIZE);
    p = emit_uncompressed_block_header(p, current_block_remaining, is_final_block);

    // walk image row by row, produce scanlines on the fly
    for (uint32_t y = 0; y < height; ++y) {
        // 1) filter byte (always 0)
        {
            // write to deflate stream
            *p++ = 0;
            // adler
            s1 += 0;
            s2 += s1;
            if (--current_block_remaining == 0) {
                remaining_bytes -= 1;
                if (remaining_bytes) {
                    current_block_remaining = (remaining_bytes > MAX_BLOCK_SIZE) ? MAX_BLOCK_SIZE : remaining_bytes;
                    is_final_block = (remaining_bytes <= MAX_BLOCK_SIZE);
                    p = emit_uncompressed_block_header(p, current_block_remaining, is_final_block);
                }
            } else {
                remaining_bytes -= 1;
            }
        }

        // 2) row data
        const unsigned char* row_src = raw_image + (size_t)y * width * 4u;
        uint32_t row_bytes_left = width * 4u;

        // We stream row_src into the deflate buffer, but we might cross a block boundary.
        while (row_bytes_left) {
            // number of bytes we can write in this block
            uint32_t can_write = (row_bytes_left < current_block_remaining) ? row_bytes_left : current_block_remaining;

            // copy contiguous chunk
            std::memcpy(p, row_src, can_write);

            // update Adler-32 for this chunk in a subloop that respects NMAX
            const unsigned char* adp = row_src;
            uint32_t chunk_left = can_write;
            while (chunk_left) {
                uint32_t t = (chunk_left > NMAX) ? NMAX : chunk_left;
                chunk_left -= t;
                for (uint32_t i = 0; i < t; ++i) {
                    s1 += *adp++;
                    s2 += s1;
                }
                s1 %= BASE;
                s2 %= BASE;
            }

            p += can_write;
            row_src += can_write;
            row_bytes_left -= can_write;
            current_block_remaining -= can_write;
            remaining_bytes -= can_write;

            // need a new block?
            if (current_block_remaining == 0 && remaining_bytes) {
                current_block_remaining = (remaining_bytes > MAX_BLOCK_SIZE) ? MAX_BLOCK_SIZE : remaining_bytes;
                is_final_block = (remaining_bytes <= MAX_BLOCK_SIZE);
                p = emit_uncompressed_block_header(p, current_block_remaining, is_final_block);
            }
        }
    }

    // finalize Adler
    s1 %= BASE;
    s2 %= BASE;
    uint32_t adler = (s2 << 16) | s1;
    write_be32(p, adler);
    p += 4;

    // write IDAT
    write_chunk(fp, "IDAT", zlib_buf, zlib_size);
    delete[] zlib_buf;

    // IEND
    write_chunk(fp, "IEND", nullptr, 0);

    fclose(fp);
}

#else

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

#endif