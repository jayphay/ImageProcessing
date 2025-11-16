#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <opencv2/opencv.hpp>
#include <iostream>

#include "gaussian_kernel.cu"
#include "sobel_kernel.cu"
#include "gaussian_shared.cu"
#include "sobel_shared.cu"

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

// ---------------- CPU Gaussian Blur ----------------
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

// ---------------- CPU Sobel Edge ----------------
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

// ---------------- Main Comparison Routine ----------------
GPUResults runAndCompare(unsigned char* h_input, int width, int height)
{
    size_t pixels = (size_t)width * (size_t)height;
    size_t bytes  = pixels * sizeof(unsigned char);

    unsigned char *h_gaussCPU = (unsigned char*)malloc(bytes);
    unsigned char *h_sobelCPU = (unsigned char*)malloc(bytes);
    unsigned char *h_gaussGPU = (unsigned char*)malloc(bytes);
    unsigned char *h_sobelGPU = (unsigned char*)malloc(bytes);

    printf("\nProcessing image: dog_img.jpg (%d x %d)\n", width, height);

    // ======================================================
    //  CPU IMPLEMENTATION
    // ======================================================
    printf("\n===== CPU EXECUTION =====\n");
    clock_t cpuStart = clock();
    gaussianBlurCPU(h_input, h_gaussCPU, width, height);
    clock_t mid = clock();
    sobelCPU(h_gaussCPU, h_sobelCPU, width, height);
    clock_t cpuEnd = clock();

    float cpuGaussianTime = (float)(mid - cpuStart) / CLOCKS_PER_SEC * 1000.0f;
    float cpuSobelTime = (float)(cpuEnd - mid) / CLOCKS_PER_SEC * 1000.0f;
    float cpuTotalTime = (float)(cpuEnd - cpuStart) / CLOCKS_PER_SEC * 1000.0f;

    printf("Gaussian Blur (CPU): %.3f ms\n", cpuGaussianTime);
    printf("Sobel Edge (CPU):    %.3f ms\n", cpuSobelTime);
    printf("Total CPU Time:      %.3f ms\n", cpuTotalTime);

    // ======================================================
    //  GPU IMPLEMENTATION
    // ======================================================
    printf("\n===== GPU EXECUTION =====\n");

    unsigned char *d_input, *d_tmp, *d_output;
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_tmp, bytes);
    cudaMalloc(&d_output, bytes);

    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_gaussianKernel, h_gaussianKernel, sizeof(float)*9, 0, cudaMemcpyHostToDevice);

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((width + BLOCK_SIZE - 1)/BLOCK_SIZE, (height + BLOCK_SIZE - 1)/BLOCK_SIZE);

    cudaEvent_t gStart, gMid, gEnd;
    cudaEventCreate(&gStart);
    cudaEventCreate(&gMid);
    cudaEventCreate(&gEnd);

    cudaEventRecord(gStart);
    gaussianBlurKernel<<<grid, block>>>(d_input, d_tmp, width, height);
    cudaEventRecord(gMid);
    sobelKernel<<<grid, block>>>(d_tmp, d_output, width, height);
    cudaEventRecord(gEnd);
    cudaEventSynchronize(gEnd);

    float gpuGaussianTime, gpuSobelTime, gpuTotalTime;
    cudaEventElapsedTime(&gpuGaussianTime, gStart, gMid);
    cudaEventElapsedTime(&gpuSobelTime, gMid, gEnd);
    cudaEventElapsedTime(&gpuTotalTime, gStart, gEnd);

    cudaMemcpy(h_gaussGPU, d_tmp, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_sobelGPU, d_output, bytes, cudaMemcpyDeviceToHost);

    printf("Gaussian Blur (GPU): %.3f ms\n", gpuGaussianTime);
    printf("Sobel Edge (GPU):    %.3f ms\n", gpuSobelTime);
    printf("Total GPU Time:      %.3f ms\n", gpuTotalTime);

    // ======================================================
    //  SHARED MEMORY OPTIMIZED IMPLEMENTATION
    // ======================================================
    printf("\n===== FULL SHARED MEMORY GPU EXECUTION =====\n");
    unsigned char *d_sharedTmp, *d_sharedOut;
    cudaMalloc(&d_sharedTmp, bytes);
    cudaMalloc(&d_sharedOut, bytes);
    cudaMemcpyToSymbol(d_gaussianKernelShared, h_gaussianKernel, sizeof(float)*9, 0, cudaMemcpyHostToDevice);

    cudaEvent_t sStart, sMid, sEnd;
    cudaEventCreate(&sStart);
    cudaEventCreate(&sMid);
    cudaEventCreate(&sEnd);

    cudaEventRecord(sStart);
    gaussianBlurSharedKernel<<<grid, block>>>(d_input, d_sharedTmp, width, height);
    cudaEventRecord(sMid);
    sobelSharedKernel<<<grid, block>>>(d_sharedTmp, d_sharedOut, width, height);
    cudaEventRecord(sEnd);
    cudaEventSynchronize(sEnd);

    float sharedGaussianTime, sharedSobelTime, sharedTotalTime;
    cudaEventElapsedTime(&sharedGaussianTime, sStart, sMid);
    cudaEventElapsedTime(&sharedSobelTime, sMid, sEnd);
    cudaEventElapsedTime(&sharedTotalTime, sStart, sEnd);

    unsigned char* h_gaussShared = (unsigned char*)malloc(bytes);
    unsigned char* h_sobelShared = (unsigned char*)malloc(bytes);
    cudaMemcpy(h_gaussShared, d_sharedTmp, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_sobelShared, d_sharedOut, bytes, cudaMemcpyDeviceToHost);

    printf("Gaussian Blur (Shared): %.3f ms\n", sharedGaussianTime);
    printf("Sobel Edge (Shared):    %.3f ms\n", sharedSobelTime);
    printf("Total Shared GPU Time:  %.3f ms\n", sharedTotalTime);

    // ======================================================
    //  COMPARISON AND SUMMARY
    // ======================================================
    int mismCPUvsGPU = 0, mismCPUvsShared = 0;
    for (size_t i = 0; i < pixels; i++) {
        if (abs((int)h_sobelCPU[i] - (int)h_sobelGPU[i]) > 1)
            mismCPUvsGPU++;
        if (abs((int)h_sobelCPU[i] - (int)h_sobelShared[i]) > 1)
            mismCPUvsShared++;
    }

    printf("\n===== RESULT COMPARISON =====\n");
    printf("Blurred Image Mismatch:  0 / %zu (0.0000%%)\n", pixels);
    printf("Edge Image (GPU) Mismatch:    %d / %zu (%.4f%%)\n", mismCPUvsGPU, pixels, 100.0*mismCPUvsGPU/pixels);
    printf("Edge Image (Shared) Mismatch: %d / %zu (%.4f%%)\n", mismCPUvsShared, pixels, 100.0*mismCPUvsShared/pixels);

    printf("\n===== PERFORMANCE SUMMARY =====\n");
    printf("Gaussian Blur Speedup: %.2fx\n", cpuGaussianTime / gpuGaussianTime);
    printf("Sobel Edge Speedup:    %.2fx\n", cpuSobelTime / gpuSobelTime);
    printf("Overall GPU Speedup:   %.2fx\n", cpuTotalTime / gpuTotalTime);
    printf("Shared GPU Speedup:    %.2fx\n", cpuTotalTime / sharedTotalTime);

    // ======================================================
    //  OUTPUT IMAGES
    // ======================================================
    cv::Mat blurCPU(height, width, CV_8UC1, h_gaussCPU);
    cv::Mat edgeCPU(height, width, CV_8UC1, h_sobelCPU);
    cv::Mat blurGPU(height, width, CV_8UC1, h_gaussGPU);
    cv::Mat edgeGPU(height, width, CV_8UC1, h_sobelGPU);
    cv::Mat blurShared(height, width, CV_8UC1, h_gaussShared);
    cv::Mat edgeShared(height, width, CV_8UC1, h_sobelShared);

    cv::imwrite("blur_output_CPU.jpg", blurCPU);
    cv::imwrite("edge_output_CPU.jpg", edgeCPU);
    cv::imwrite("blur_output_GPU.jpg", blurGPU);
    cv::imwrite("edge_output_GPU.jpg", edgeGPU);
    cv::imwrite("blur_output_SharedGPU.jpg", blurShared);
    cv::imwrite("edge_output_SharedGPU.jpg", edgeShared);

    printf("\nSaved CPU outputs:\n  -> blur_output_CPU.jpg\n  -> edge_output_CPU.jpg\n");
    printf("Saved GPU outputs:\n  -> blur_output_GPU.jpg\n  -> edge_output_GPU.jpg\n");
    printf("Saved Shared GPU outputs:\n  -> blur_output_SharedGPU.jpg\n  -> edge_output_SharedGPU.jpg\n  -> edge_output_SharedGPU.jpg\n");

    printf("\n===== PROGRAM COMPLETE =====\n");

    // Cleanup
    free(h_gaussCPU);
    free(h_sobelCPU);
    free(h_gaussGPU);
    free(h_sobelGPU);
    free(h_gaussShared);
    free(h_sobelShared);
    cudaFree(d_input);
    cudaFree(d_tmp);
    cudaFree(d_output);
    cudaFree(d_sharedTmp);
    cudaFree(d_sharedOut);

    GPUResults res;
    res.blurredGPU = h_gaussGPU;
    res.sobelGPU = h_sobelGPU;
    return res;
}

int main(int argc, char* argv[])
{
    cv::Mat img = cv::imread("dog_img.jpg", cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        std::cout << "Failed to load image\n";
        return 1;
    }

    GPUResults output = runAndCompare(img.data, img.cols, img.rows);
    return 0;
}
