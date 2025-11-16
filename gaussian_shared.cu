#include <cuda_runtime.h>
#include <stdio.h>

#define BLOCK_SIZE 16
__constant__ float d_gaussianKernelShared[9] = {
    1/16.0f, 2/16.0f, 1/16.0f,
    2/16.0f, 4/16.0f, 2/16.0f,
    1/16.0f, 2/16.0f, 1/16.0f
};

// Shared memory optimized Gaussian blur
__global__ void gaussianBlurSharedKernel(unsigned char* input, unsigned char* output, int width, int height)
{
    __shared__ unsigned char tile[BLOCK_SIZE + 2][BLOCK_SIZE + 2];

    int x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int y = blockIdx.y * BLOCK_SIZE + threadIdx.y;

    int tx = threadIdx.x + 1;
    int ty = threadIdx.y + 1;

    if (x < width && y < height)
        tile[ty][tx] = input[y * width + x];

    // Halo loading
    if (threadIdx.x == 0 && x > 0)
        tile[ty][0] = input[y * width + x - 1];
    if (threadIdx.x == BLOCK_SIZE - 1 && x < width - 1)
        tile[ty][tx + 1] = input[y * width + x + 1];
    if (threadIdx.y == 0 && y > 0)
        tile[0][tx] = input[(y - 1) * width + x];
    if (threadIdx.y == BLOCK_SIZE - 1 && y < height - 1)
        tile[ty + 1][tx] = input[(y + 1) * width + x];

    __syncthreads();

    if (x < width && y < height) {
        float sum = 0.0f;
        for (int ky = -1; ky <= 1; ky++) {
            for (int kx = -1; kx <= 1; kx++) {
                sum += tile[ty + ky][tx + kx] *
                        d_gaussianKernelShared[(ky + 1) * 3 + (kx + 1)];
            }
        }
        output[y * width + x] = (unsigned char)sum;
    }
}
