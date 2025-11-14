#define N_BLOCKS_PER_SM 2
#define N_THREADS_PER_BLOCK 256
#define N_SMS 80  // adjust according to your GPU

#define N_BLOCKS (((0xffffffffll) + N_THREADS_PER_BLOCK) / N_THREADS_PER_BLOCK)