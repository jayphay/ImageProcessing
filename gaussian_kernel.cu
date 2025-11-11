#include <cuda_runtime.h>
#include <stdio.h>

#define BLOCK_SIZE 16

// 3x3 Gaussian Kernel
__constant__ float d_gaussianKernel[9] = {
    1/16.0f, 2/16.0f, 1/16.0f,
    2/16.0f, 4/16.0f, 2/16.0f,
    1/16.0f, 2/16.0f, 1/16.0f
};

__global__ void gaussianBlurKernel(unsigned char* input, unsigned char* output, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    float sum = 0.0f;

    // Apply 3x3 Gaussian filter
    for (int ky = -1; ky <= 1; ky++) {
        for (int kx = -1; kx <= 1; kx++) {
            int px = min(max(x + kx, 0), width - 1);
            int py = min(max(y + ky, 0), height - 1);
            float pixel = input[py * width + px];
            float weight = d_gaussianKernel[(ky+1)*3 + (kx+1)];
            sum += pixel * weight;
        }
    }
    output[y * width + x] = (unsigned char)sum;
}
