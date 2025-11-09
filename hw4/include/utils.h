#include <cassert>
#include <string>

// convert one hex-codec char to binary
unsigned char decode(unsigned char c);

// convert hex string to binary
//
// in: input string
// string_len: the length of the input string
//      '\0' is not included in string_len!!!
// out: output bytes array
void convert_string_to_little_endian_bytes(unsigned char* out, char* in, size_t string_len);

int little_endian_bit_comparison(const unsigned char* a, const unsigned char* b, size_t byte_len);

// print out binary array (from highest value) in the hex format
void print_hex(unsigned char* hex, size_t len);
// print out binar array (from lowest value) in the hex format
void print_hex_inverse(unsigned char* hex, size_t len);

void getline(char* str, size_t len, FILE* fp);
