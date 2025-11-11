__global__ void sobelKernel(unsigned char* input, unsigned char* output, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x <= 0 || x >= width-1 || y <= 0 || y >= height-1) return;

    int Gx =
        -input[(y-1)*width + (x-1)] - 2*input[y*width + (x-1)] - input[(y+1)*width + (x-1)] +
         input[(y-1)*width + (x+1)] + 2*input[y*width + (x+1)] + input[(y+1)*width + (x+1)];

    int Gy =
         input[(y-1)*width + (x-1)] + 2*input[(y-1)*width + x] + input[(y-1)*width + (x+1)] -
         input[(y+1)*width + (x-1)] - 2*input[(y+1)*width + x] - input[(y+1)*width + (x+1)];

    int mag = (int)sqrtf((float)(Gx * Gx + Gy * Gy));
    output[y * width + x] = (mag > 255 ? 255 : mag);
}