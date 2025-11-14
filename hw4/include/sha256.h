#ifndef __SHA256_HEADER__
#define __SHA256_HEADER__

#include <stddef.h>

//--------------- DATA TYPES --------------
typedef unsigned int WORD;
typedef unsigned char BYTE;

typedef union _sha256_ctx {
    WORD h[8];
    BYTE b[32];
} SHA256;

//----------- FUNCTION DECLARATION --------
namespace device {

__device__ void sha256_transform(SHA256* ctx, const BYTE* msg);
__device__ void sha256(SHA256* ctx, const BYTE* msg, size_t len);

}  // namespace device

namespace host {

void sha256_transform(SHA256* ctx, const BYTE* msg);
void sha256(SHA256* ctx, const BYTE* msg, size_t len);

}  // namespace host

#endif  //__SHA256_HEADER__
