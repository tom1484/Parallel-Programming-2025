#include <cstring>
#include <string>

#include "hash.h"
#include "utils.h"

namespace device {

__device__ void double_sha256(SHA256* sha256_ctx, unsigned char* bytes, size_t len) {
    SHA256 tmp;
    sha256_80(&tmp, (BYTE*)bytes, len);
    sha256_64(sha256_ctx, (BYTE*)&tmp, sizeof(tmp));
}

}  // namespace device

namespace host {

void double_sha256(SHA256* sha256_ctx, unsigned char* bytes, size_t len) {
    SHA256 tmp;
    sha256(&tmp, (BYTE*)bytes, len);
    sha256(sha256_ctx, (BYTE*)&tmp, sizeof(tmp));
}

void calc_merkle_root(unsigned char* root, int count, char** branch) {
    size_t total_count = count;  // merkle branch
    unsigned char* raw_list = new unsigned char[(total_count + 1) * 32];
    unsigned char** list = new unsigned char*[total_count + 1];

    // copy each branch to the list
    for (int i = 0; i < total_count; ++i) {
        list[i] = raw_list + i * 32;
        // convert hex string to bytes array and store them into the list
        convert_string_to_little_endian_bytes(list[i], branch[i], 64);
    }

    list[total_count] = raw_list + total_count * 32;

    // calculate merkle root
    while (total_count > 1) {
        // hash each pair
        int i, j;

        if (total_count % 2 == 1)  // odd,
        {
            memcpy(list[total_count], list[total_count - 1], 32);
        }

        for (i = 0, j = 0; i < total_count; i += 2, ++j) {
            // this part is slightly tricky,
            //   because of the implementation of the double_sha256,
            //   we can avoid the memory begin overwritten during our sha256d calculation
            // double_sha:
            //     tmp = hash(list[0]+list[1])
            //     list[0] = hash(tmp)
            double_sha256((SHA256*)list[j], list[i], 64);
        }

        total_count = j;
    }

    memcpy(root, list[0], 32);

    delete[] raw_list;
    delete[] list;
}

}  // namespace host
