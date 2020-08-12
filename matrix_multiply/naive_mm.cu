#include <iostream>
#include "utils.h"
using namespace std;


#define N 32


__global__ void naive_mm(float* ma, float* mb, float* mc, int n) {
    for (int i = 0 + threadIdx.x; i < n; i += blockDim.x) {
        for (int j = 0 + threadIdx.y; j < n; j += threadIdx.y) {
            for (int k = 0; k < n; ++k) {
                mc[i * n + j] += ma[i * n + k] * mb[k * n + j];
            }
        }
    }
}


int main(int argc, char** argv) {
    float* ha = new float[N * N];
    float* hb = new float[N * N];
    float* hc = new float[N * N];

    float *da, *db, *dc;


    for (int i = 0; i < N * N; ++i) ha[i] = hb[i] = i;

    CUDA_CALL(cudaMalloc(&da, sizeof(float) * N * N));
    CUDA_CALL(cudaMalloc(&db, sizeof(float) * N * N));
    CUDA_CALL(cudaMalloc(&dc, sizeof(float) * N * N));

    CUDA_CALL(cudaMemcpy(da, ha, sizeof(float) * N * N, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(db, hb, sizeof(float) * N * N, cudaMemcpyHostToDevice));

    float time;
    cudaEvent_t start, stop;
    CUDA_CALL(cudaEventCreate(&start));
    CUDA_CALL(cudaEventCreate(&stop));

    CUDA_CALL(cudaEventRecord(start, 0));
    naive_mm<<<1, 1024>>>(da, db, dc, N);
    CUDA_CALL(cudaEventRecord(stop, 0));
    CUDA_CALL(cudaEventSynchronize(stop));
    CUDA_CALL(cudaEventElapsedTime(&time, start, stop));
    printf("Time: %f ms\n", time);
    
    CUDA_CALL(cudaMemcpy(hc, dc, sizeof(float) * N * N, cudaMemcpyDeviceToHost));

    CUDA_CALL(cudaFree(da));
    CUDA_CALL(cudaFree(db));
    CUDA_CALL(cudaFree(dc));

    delete[] ha;
    delete[] hb;
    delete[] hc;
}