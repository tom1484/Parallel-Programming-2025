#include "utils.h"

unsigned char decode(unsigned char c) {
    switch (c) {
        case 'a':
            return 0x0a;
        case 'b':
            return 0x0b;
        case 'c':
            return 0x0c;
        case 'd':
            return 0x0d;
        case 'e':
            return 0x0e;
        case 'f':
            return 0x0f;
        case '0' ... '9':
            return c - '0';
    }
}

void convert_string_to_little_endian_bytes(unsigned char* out, char* in, size_t string_len) {
    assert(string_len % 2 == 0);

    size_t s = 0;
    size_t b = string_len / 2 - 1;

    for (s, b; s < string_len; s += 2, --b) {
        out[b] = (unsigned char)(decode(in[s]) << 4) + decode(in[s + 1]);
    }
}

__host__ __device__ int little_endian_bit_comparison(const unsigned char* a, const unsigned char* b, size_t byte_len) {
    // compared from lowest bit
    for (int i = byte_len - 1; i >= 0; --i) {
        if (a[i] < b[i])
            return -1;
        else if (a[i] > b[i])
            return 1;
    }
    return 0;
}

void print_hex(unsigned char* hex, size_t len) {
    for (int i = 0; i < len; ++i) {
        PRINTF("%02x", hex[i]);
    }
}

void print_hex_inverse(unsigned char* hex, size_t len) {
    for (int i = len - 1; i >= 0; --i) {
        PRINTF("%02x", hex[i]);
    }
}

void getline(char* str, size_t len, FILE* fp) {
    int i = 0;
    while (i < len && (str[i] = fgetc(fp)) != EOF && str[i++] != '\n');
    str[len - 1] = '\0';
}
