#ifndef SUBMIT
#include <chrono>
#include <iostream>
#endif

#include "hash.h"
#include "sha256.h"
#include "utils.h"

using namespace std;

void solve(FILE* fin, FILE* fout) {
    // **** read data *****
    char version[9];
    char prevhash[65];
    char ntime[9];
    char nbits[9];
    int tx;
    char* raw_merkle_branch;
    char** merkle_branch;

    getline(version, 9, fin);
    getline(prevhash, 65, fin);
    getline(ntime, 9, fin);
    getline(nbits, 9, fin);
    fscanf(fin, "%d\n", &tx);
    PRINTF("start hashing");

    raw_merkle_branch = new char[tx * 65];
    merkle_branch = new char*[tx];
    for (int i = 0; i < tx; ++i) {
        merkle_branch[i] = raw_merkle_branch + i * 65;
        getline(merkle_branch[i], 65, fin);
        merkle_branch[i][64] = '\0';
    }

    // **** calculate merkle root ****

    unsigned char merkle_root[32];
    calc_merkle_root(merkle_root, tx, merkle_branch);

    PRINTF("merkle root(little): ");
    print_hex(merkle_root, 32);
    PRINTF("\n");

    PRINTF("merkle root(big):    ");
    print_hex_inverse(merkle_root, 32);
    PRINTF("\n");

    // **** solve block ****
    PRINTF("Block info (big): \n");
    PRINTF("  version:  %s\n", version);
    PRINTF("  pervhash: %s\n", prevhash);
    PRINTF("  merkleroot: ");
    print_hex_inverse(merkle_root, 32);
    PRINTF("\n");
    PRINTF("  nbits:    %s\n", nbits);
    PRINTF("  ntime:    %s\n", ntime);
    PRINTF("  nonce:    ???\n\n");

    HashBlock block;

    // convert to byte array in little-endian
    convert_string_to_little_endian_bytes((unsigned char*)&block.version, version, 8);
    convert_string_to_little_endian_bytes(block.prevhash, prevhash, 64);
    memcpy(block.merkle_root, merkle_root, 32);
    convert_string_to_little_endian_bytes((unsigned char*)&block.nbits, nbits, 8);
    convert_string_to_little_endian_bytes((unsigned char*)&block.ntime, ntime, 8);
    block.nonce = 0;

    // ********** calculate target value *********
    // calculate target value from encoded difficulty which is encoded on "nbits"
    unsigned int exp = block.nbits >> 24;
    unsigned int mant = block.nbits & 0xffffff;
    unsigned char target_hex[32] = {};

    unsigned int shift = 8 * (exp - 3);
    unsigned int sb = shift / 8;
    unsigned int rb = shift % 8;

    // little-endian
    target_hex[sb] = (mant << rb);
    target_hex[sb + 1] = (mant >> (8 - rb));
    target_hex[sb + 2] = (mant >> (16 - rb));
    target_hex[sb + 3] = (mant >> (24 - rb));

    PRINTF("Target value (big): ");
    print_hex_inverse(target_hex, 32);
    PRINTF("\n");

    // ********** find nonce **************

    SHA256 sha256_ctx;

    for (block.nonce = 0x00000000; block.nonce <= 0xffffffff; ++block.nonce) {
        // sha256d
        double_sha256(&sha256_ctx, (unsigned char*)&block, sizeof(block));
        if (block.nonce % 1000000 == 0) {
            PRINTF("hash #%10u (big): ", block.nonce);
            print_hex_inverse(sha256_ctx.b, 32);
            PRINTF("\n");
        }

        if (little_endian_bit_comparison(sha256_ctx.b, target_hex, 32) < 0)  // sha256_ctx < target_hex
        {
            PRINTF("Found Solution!!\n");
            PRINTF("hash #%10u (big): ", block.nonce);
            print_hex_inverse(sha256_ctx.b, 32);
            PRINTF("\n\n");

            break;
        }
    }

    // print result

    // little-endian
    PRINTF("hash(little): ");
    print_hex(sha256_ctx.b, 32);
    PRINTF("\n");

    // big-endian
    PRINTF("hash(big):    ");
    print_hex_inverse(sha256_ctx.b, 32);
    PRINTF("\n\n");

    for (int i = 0; i < 4; ++i) {
        fprintf(fout, "%02x", ((unsigned char*)&block.nonce)[i]);
    }
    fprintf(fout, "\n");

    delete[] merkle_branch;
    delete[] raw_merkle_branch;
}

int main(int argc, char** argv) {
#ifndef SUBMIT
    auto __start = chrono::high_resolution_clock::now();
#endif

    if (argc != 3) {
        EPRINTF("usage: cuda_miner <in> <out>\n");
    }
    FILE* fin = fopen(argv[1], "r");
    FILE* fout = fopen(argv[2], "w");

    int totalblock;

    fscanf(fin, "%d\n", &totalblock);
    fprintf(fout, "%d\n", totalblock);
    
    PRINTF("Total blocks to solve: %d\n", totalblock);

    for (int i = 0; i < totalblock; ++i) {
        solve(fin, fout);
    }

#ifndef SUBMIT
    auto __end = chrono::high_resolution_clock::now();
    auto __elapsed_us = chrono::duration_cast<chrono::microseconds>(__end - __start);
    cerr << "Elapsed: " << __elapsed_us.count() << " us" << endl;
#endif

    return 0;
}