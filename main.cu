#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include "gaussian_kernel.cu"
#include "sobel_kernel.cu"
#include "readPgm.c"
#include "writePgm.c"

#define BLOCK_SIZE 16
typedef struct {
    unsigned char* blurredGPU;
    unsigned char* sobelGPU;
} GPUResults;

float h_gaussianKernel[9] = {
    1/16.0f, 2/16.0f, 1/16.0f,
    2/16.0f, 4/16.0f, 2/16.0f,
    1/16.0f, 2/16.0f, 1/16.0f
};

void gaussianBlurCPU(unsigned char* input, unsigned char* output, int width, int height)
{
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float sum = 0.0f;
            for (int ky = -1; ky <= 1; ky++) {
                for (int kx = -1; kx <= 1; kx++) {
                    int px = fmin(fmax(x + kx, 0), width - 1);
                    int py = fmin(fmax(y + ky, 0), height - 1);
                    sum += input[py * width + px] *
                           h_gaussianKernel[(ky+1)*3 + (kx+1)];
                }
            }
            output[y * width + x] = (unsigned char)sum;
        }
    }
}

void sobelCPU(unsigned char* input, unsigned char* output, int width, int height)
{
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {

            int Gx =
                -input[(y-1)*width + (x-1)] - 2*input[y*width + (x-1)] - input[(y+1)*width + (x-1)] +
                 input[(y-1)*width + (x+1)] + 2*input[y*width + (x+1)] + input[(y+1)*width + (x+1)];

            int Gy =
                 input[(y-1)*width + (x-1)] + 2*input[(y-1)*width + x] + input[(y-1)*width + (x+1)] -
                 input[(y+1)*width + (x-1)] - 2*input[(y+1)*width + x] - input[(y+1)*width + (x+1)];

            int mag = (int)sqrtf((float)(Gx * Gx + Gy * Gy));
            output[y * width + x] = (mag > 255 ? 255 : mag);
        }
    }
}

// ...existing code...
GPUResults runAndCompare(unsigned char* h_input, int width, int height)
{
    size_t pixels = (size_t)width * (size_t)height;
    size_t bytes  = pixels * sizeof(unsigned char);

    unsigned char *h_gaussCPU = (unsigned char*)malloc(bytes);
    unsigned char *h_sobelCPU = (unsigned char*)malloc(bytes);
    unsigned char *h_gaussGPU = (unsigned char*)malloc(bytes);
    unsigned char *h_sobelGPU = (unsigned char*)malloc(bytes);

    // ---- CPU Timing ----
    clock_t start = clock();
    gaussianBlurCPU(h_input, h_gaussCPU, width, height);
    sobelCPU(h_gaussCPU, h_sobelCPU, width, height);
    float cpuTime = (float)(clock() - start) / CLOCKS_PER_SEC * 1000.0f;

    // ---- GPU Memory ----
    unsigned char *d_input = nullptr, *d_tmp = nullptr, *d_output = nullptr;
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_tmp, bytes);
    cudaMalloc(&d_output, bytes);

    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_gaussianKernel, h_gaussianKernel, sizeof(float)*9, 0, cudaMemcpyHostToDevice);

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((width + BLOCK_SIZE - 1)/BLOCK_SIZE,
              (height + BLOCK_SIZE - 1)/BLOCK_SIZE);

    // ---- GPU Timing ----
    cudaEvent_t startEvt, stopEvt;
    cudaEventCreate(&startEvt);
    cudaEventCreate(&stopEvt);
    cudaEventRecord(startEvt);

    gaussianBlurKernel<<<grid, block>>>(d_input, d_tmp, width, height);
    sobelKernel<<<grid, block>>>(d_tmp, d_output, width, height);

    cudaEventRecord(stopEvt);
    cudaEventSynchronize(stopEvt);

    float gpuTime;
    cudaEventElapsedTime(&gpuTime, startEvt, stopEvt);

    // Copy results back
    cudaMemcpy(h_gaussGPU, d_tmp, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_sobelGPU, d_output, bytes, cudaMemcpyDeviceToHost);

    printf("CPU Time: %.3f ms\n", cpuTime);
    printf("GPU Time: %.3f ms\n", gpuTime);

    // ---- Compare correctness ----
    int mismatch = 0;
    for (size_t i = 0; i < pixels; i++)
        if (abs((int)h_sobelCPU[i] - (int)h_sobelGPU[i]) > 1)
            mismatch++;

    printf("Mismatched pixels: %d / %zu\n", mismatch, pixels);

    GPUResults results;
    results.blurredGPU = h_gaussGPU;
    results.sobelGPU   = h_sobelGPU;

    // free temporary / CPU buffers, but NOT the GPU result host buffers
    cudaFree(d_input);
    cudaFree(d_tmp);
    cudaFree(d_output);
    free(h_gaussCPU);
    free(h_sobelCPU);

    return results;
}

int main(int argc, char* argv[]) {
    cv::Mat img = cv::imread("dog_img.jpg", cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        std::cout << "Failed to load image\n";
        return 1;
    }
    if (img.channels() != 1) {
        cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
    }

    GPUResults output = runAndCompare(img.data, img.cols, img.rows);

    // create Mats that wrap the returned buffers (no copy) then write
    cv::Mat blurImg(img.rows, img.cols, CV_8UC1, output.blurredGPU);
    cv::Mat edgeImg(img.rows, img.cols, CV_8UC1, output.sobelGPU);

    cv::imwrite("blur_output.jpg", blurImg);
    cv::imwrite("edge_output.jpg", edgeImg);

    // free the returned buffers after writing files
    free(output.blurredGPU);
    free(output.sobelGPU);
    return 0;
}
// ...existing code...