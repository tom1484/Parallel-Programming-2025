#include <cassert>
#include <chrono>
#include <string>

#ifdef DEBUG
#define PRINTF(...) printf(__VA_ARGS__)
#define EPRINTF(...) fprintf(std::stderr, __VA_ARGS__)
#else
#define PRINTF(...) ((void)0)
#define EPRINTF(...) ((void)0)
#endif

#define PROFILE(name) Profiler __profiler(name)

class Profiler {
   public:
    Profiler() {}
    Profiler(const char* name) : name(name), start(std::chrono::high_resolution_clock::now()) {}
    ~Profiler() {
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        PRINTF("PROFILE [%s]: %ld us\n", name, elapsed.count());
    }

   private:
    const char* name;
    std::chrono::high_resolution_clock::time_point start;
};

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
