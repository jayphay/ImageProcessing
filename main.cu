#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include "gaussian_kernel.cu"
#include "sobel_kernel.cu"

#define BLOCK_SIZE 16

typedef struct {
    unsigned char* blurredGPU;
    unsigned char* sobelGPU;
} GPUResults;

// Host Gaussian kernel
float h_gaussianKernel[9] = {
    1/16.0f, 2/16.0f, 1/16.0f,
    2/16.0f, 4/16.0f, 2/16.0f,
    1/16.0f, 2/16.0f, 1/16.0f
};

// ---------------- CPU FILTERS ----------------
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

// ---------------- RUN + COMPARE ----------------
GPUResults runAndCompare(unsigned char* h_input, int width, int height)
{
    size_t pixels = (size_t)width * (size_t)height;
    size_t bytes  = pixels * sizeof(unsigned char);

    unsigned char *h_gaussCPU = (unsigned char*)malloc(bytes);
    unsigned char *h_sobelCPU = (unsigned char*)malloc(bytes);
    unsigned char *h_gaussGPU = (unsigned char*)malloc(bytes);
    unsigned char *h_sobelGPU = (unsigned char*)malloc(bytes);

    printf("\n===== CPU EXECUTION =====\n");

    // --- Gaussian Blur (CPU)
    clock_t gStart = clock();
    gaussianBlurCPU(h_input, h_gaussCPU, width, height);
    clock_t gEnd = clock();
    float gaussCPUTime = (float)(gEnd - gStart) / CLOCKS_PER_SEC * 1000.0f;

    // --- Sobel Edge Detection (CPU)
    clock_t sStart = clock();
    sobelCPU(h_gaussCPU, h_sobelCPU, width, height);
    clock_t sEnd = clock();
    float sobelCPUTime = (float)(sEnd - sStart) / CLOCKS_PER_SEC * 1000.0f;

    float totalCPUTime = gaussCPUTime + sobelCPUTime;

    printf("Gaussian Blur (CPU): %.3f ms\n", gaussCPUTime);
    printf("Sobel Edge (CPU):    %.3f ms\n", sobelCPUTime);
    printf("Total CPU Time:      %.3f ms\n", totalCPUTime);

    // ---------------- GPU ----------------
    printf("\n===== GPU EXECUTION =====\n");
    unsigned char *d_input = nullptr, *d_tmp = nullptr, *d_output = nullptr;
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_tmp, bytes);
    cudaMalloc(&d_output, bytes);

    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_gaussianKernel, h_gaussianKernel, sizeof(float)*9, 0, cudaMemcpyHostToDevice);

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((width + BLOCK_SIZE - 1)/BLOCK_SIZE,
              (height + BLOCK_SIZE - 1)/BLOCK_SIZE);

    cudaEvent_t startEvt, midEvt, stopEvt;
    cudaEventCreate(&startEvt);
    cudaEventCreate(&midEvt);
    cudaEventCreate(&stopEvt);

    cudaEventRecord(startEvt);
    gaussianBlurKernel<<<grid, block>>>(d_input, d_tmp, width, height);
    cudaEventRecord(midEvt);
    sobelKernel<<<grid, block>>>(d_tmp, d_output, width, height);
    cudaEventRecord(stopEvt);
    cudaEventSynchronize(stopEvt);

    float totalGPUTime, blurGPUTime, sobelGPUTime;
    cudaEventElapsedTime(&totalGPUTime, startEvt, stopEvt);
    cudaEventElapsedTime(&blurGPUTime, startEvt, midEvt);
    cudaEventElapsedTime(&sobelGPUTime, midEvt, stopEvt);

    cudaMemcpy(h_gaussGPU, d_tmp, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_sobelGPU, d_output, bytes, cudaMemcpyDeviceToHost);

    printf("Gaussian Blur (GPU): %.3f ms\n", blurGPUTime);
    printf("Sobel Edge (GPU):    %.3f ms\n", sobelGPUTime);
    printf("Total GPU Time:      %.3f ms\n", totalGPUTime);

    // ---------------- Comparison ----------------
    int mismBlur = 0, mismEdge = 0;
    for (size_t i = 0; i < pixels; i++) {
        if (abs((int)h_gaussCPU[i] - (int)h_gaussGPU[i]) > 1) mismBlur++;
        if (abs((int)h_sobelCPU[i] - (int)h_sobelGPU[i]) > 1) mismEdge++;
    }

    printf("\n===== RESULT COMPARISON =====\n");
    printf("Blurred Image Mismatch: %d / %zu (%.4f%%)\n", mismBlur, pixels, 100.0*mismBlur/pixels);
    printf("Edge Image Mismatch:    %d / %zu (%.4f%%)\n", mismEdge, pixels, 100.0*mismEdge/pixels);

    printf("\n===== PERFORMANCE SUMMARY =====\n");
    printf("Gaussian Blur Speedup: %.2fx\n", gaussCPUTime / blurGPUTime);
    printf("Sobel Edge Speedup:    %.2fx\n", sobelCPUTime / sobelGPUTime);
    printf("Overall Speedup:       %.2fx\n", totalCPUTime / totalGPUTime);

    // ---------------- Cleanup ----------------
    cudaFree(d_input);
    cudaFree(d_tmp);
    cudaFree(d_output);
    free(h_gaussCPU);
    free(h_sobelCPU);

    GPUResults results;
    results.blurredGPU = h_gaussGPU;
    results.sobelGPU   = h_sobelGPU;
    return results;
}

// ---------------- MAIN ----------------
int main(int argc, char* argv[])
{
    std::string filename = (argc > 1) ? argv[1] : "dog_img.jpg";
    cv::Mat img = cv::imread(filename, cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        std::cerr << "Failed to load image: " << filename << "\n";
        return 1;
    }

    printf("\nProcessing image: %s (%d x %d)\n", filename.c_str(), img.cols, img.rows);

    GPUResults output = runAndCompare(img.data, img.cols, img.rows);

    // Save outputs
    cv::Mat blurImg(img.rows, img.cols, CV_8UC1, output.blurredGPU);
    cv::Mat edgeImg(img.rows, img.cols, CV_8UC1, output.sobelGPU);
    cv::imwrite("blur_output_GPU.jpg", blurImg);
    cv::imwrite("edge_output_GPU.jpg", edgeImg);

    printf("\nSaved GPU outputs:\n  -> blur_output_GPU.jpg\n  -> edge_output_GPU.jpg\n");

    // Re-run CPU filters for saved reference outputs
    unsigned char* tmpCPU = (unsigned char*)malloc(img.total());
    unsigned char* tmpEdge = (unsigned char*)malloc(img.total());
    gaussianBlurCPU(img.data, tmpCPU, img.cols, img.rows);
    sobelCPU(tmpCPU, tmpEdge, img.cols, img.rows);
    cv::imwrite("blur_output_CPU.jpg", cv::Mat(img.rows, img.cols, CV_8UC1, tmpCPU));
    cv::imwrite("edge_output_CPU.jpg", cv::Mat(img.rows, img.cols, CV_8UC1, tmpEdge));
    free(tmpCPU);
    free(tmpEdge);

    printf("Saved CPU outputs:\n  -> blur_output_CPU.jpg\n  -> edge_output_CPU.jpg\n");

    free(output.blurredGPU);
    free(output.sobelGPU);

    printf("\n===== PROGRAM COMPLETE =====\n\n");
    return 0;
}
