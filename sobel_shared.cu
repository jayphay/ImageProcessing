#include <cuda_runtime.h>
#include <stdio.h>

#define BLOCK_SIZE 16

// Shared-memory optimized Sobel kernel
__global__ void sobelSharedKernel(unsigned char* input, unsigned char* output, int width, int height)
{
    __shared__ unsigned char tile[BLOCK_SIZE + 2][BLOCK_SIZE + 2];

    int x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int y = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int tx = threadIdx.x + 1;
    int ty = threadIdx.y + 1;

    // Load the central pixel
    if (x < width && y < height)
        tile[ty][tx] = input[y * width + x];

    // Load halo pixels
    if (threadIdx.x == 0 && x > 0)
        tile[ty][0] = input[y * width + x - 1];
    if (threadIdx.x == BLOCK_SIZE - 1 && x < width - 1)
        tile[ty][tx + 1] = input[y * width + x + 1];
    if (threadIdx.y == 0 && y > 0)
        tile[0][tx] = input[(y - 1) * width + x];
    if (threadIdx.y == BLOCK_SIZE - 1 && y < height - 1)
        tile[ty + 1][tx] = input[(y + 1) * width + x];

    // Fill corner halos
    if (threadIdx.x == 0 && threadIdx.y == 0 && x > 0 && y > 0)
        tile[0][0] = input[(y - 1) * width + x - 1];
    if (threadIdx.x == BLOCK_SIZE - 1 && threadIdx.y == 0 && x < width - 1 && y > 0)
        tile[0][tx + 1] = input[(y - 1) * width + x + 1];
    if (threadIdx.x == 0 && threadIdx.y == BLOCK_SIZE - 1 && x > 0 && y < height - 1)
        tile[ty + 1][0] = input[(y + 1) * width + x - 1];
    if (threadIdx.x == BLOCK_SIZE - 1 && threadIdx.y == BLOCK_SIZE - 1 && x < width - 1 && y < height - 1)
        tile[ty + 1][tx + 1] = input[(y + 1) * width + x + 1];

    __syncthreads();

    if (x > 0 && y > 0 && x < width - 1 && y < height - 1) {
        int Gx =
            -tile[ty - 1][tx - 1] - 2 * tile[ty][tx - 1] - tile[ty + 1][tx - 1] +
             tile[ty - 1][tx + 1] + 2 * tile[ty][tx + 1] + tile[ty + 1][tx + 1];

        int Gy =
             tile[ty - 1][tx - 1] + 2 * tile[ty - 1][tx] + tile[ty - 1][tx + 1] -
             tile[ty + 1][tx - 1] - 2 * tile[ty + 1][tx] - tile[ty + 1][tx + 1];

        int mag = (int)sqrtf((float)(Gx * Gx + Gy * Gy));
        output[y * width + x] = (mag > 255 ? 255 : mag);
    }
}
