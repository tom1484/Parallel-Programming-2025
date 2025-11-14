#ifndef __SHA256_HEADER__
#define __SHA256_HEADER__

#include <stddef.h>

#define DEFINE_H \
    WORD h0, WORD h1, WORD h2, WORD h3, \
    WORD h4, WORD h5, WORD h6, WORD h7
#define PASS_H \
    h0, h1, h2, h3, \
    h4, h5, h6, h7

//--------------- DATA TYPES --------------
typedef unsigned int WORD;
typedef unsigned char BYTE;

__align__(64) typedef union _sha256_ctx {
    WORD h[8];
    BYTE b[32];
} SHA256;

//----------- FUNCTION DECLARATION --------
namespace device {

__device__ __forceinline__ WORD rotr(WORD x, int n);
__device__ __forceinline__ void sha256_transform(SHA256* ctx, const BYTE* msg);
__device__ void sha256_32(SHA256* ctx, const BYTE* msg, size_t len);
__device__ void sha256_80(SHA256* ctx, const BYTE* msg, size_t len);

}  // namespace device

namespace host {

WORD u32be(const BYTE* b);
void sha256_transform(SHA256* ctx, const BYTE* msg);
void sha256(SHA256* ctx, const BYTE* msg, size_t len);

}  // namespace host

#endif  //__SHA256_HEADER__
