// PLEASE MODIFY THIS FILE TO IMPLEMENT YOUR SOLUTION

#include "apsp.h"

namespace {

#define BLOCK_SIZE 32
#define DATA_RANGE 100000
#define graph(i, j) graph[(i) * n + (j)]
#define shared(i, j) shared[(i) * blockDim.x + (j)]
#define shared_offset(i, j, offset) shared[(i) * blockDim.x + (j) + offset]

__global__ void stage1(int n, int p, int *graph) {
    extern __shared__ int shared[];
    auto i = p * blockDim.y + threadIdx.y;
    auto j = p * blockDim.x + threadIdx.x;
    if (i < n && j < n) shared(threadIdx.y, threadIdx.x) = graph(i, j);
    else shared(threadIdx.y, threadIdx.x) = DATA_RANGE;
    __syncthreads();
    # pragma unroll 32
    for (int k = 0; k < BLOCK_SIZE; k++) {
        shared(threadIdx.y, threadIdx.x) = min(shared(threadIdx.y, threadIdx.x), shared(threadIdx.y, k) + shared(k, threadIdx.x));
    }
    if (i < n && j < n) graph(i, j) = shared(threadIdx.y, threadIdx.x);
}

__global__ void stage2(int n, int p, int *graph) {
    extern __shared__ int shared[];
    auto i = (blockIdx.y ? p : (blockIdx.x < p ? blockIdx.x : blockIdx.x + 1)) * blockDim.y + threadIdx.y;
    auto j = (blockIdx.y ? (blockIdx.x < p ? blockIdx.x : blockIdx.x + 1) : p) * blockDim.x + threadIdx.x;
    auto ii = p * blockDim.y + threadIdx.y;
    auto jj = p * blockDim.x + threadIdx.x;
    auto blockSize = blockDim.x * blockDim.y;
    if (i < n && j < n) shared(threadIdx.y, threadIdx.x) = graph(i, j);
    else shared(threadIdx.y, threadIdx.x) = DATA_RANGE;
    if (ii < n && jj < n) shared_offset(threadIdx.y, threadIdx.x, blockSize) = graph(ii, jj);
    else shared_offset(threadIdx.y, threadIdx.x, blockSize) = DATA_RANGE;
    __syncthreads();
    # pragma unroll 32
    for (int k = 0; k < BLOCK_SIZE; k++) {
        shared(threadIdx.y, threadIdx.x) = min(shared(threadIdx.y, threadIdx.x), shared_offset(threadIdx.y, k, (blockIdx.y ? blockSize : 0)) + shared_offset(k, threadIdx.x, (blockIdx.y ? 0 : blockSize)));
    }
    if (i < n && j < n) graph(i, j) = shared(threadIdx.y, threadIdx.x);
}

__global__ void stage3(int n, int p, int *graph) {
    extern __shared__ int shared[];
    auto i = (blockIdx.y < p ? blockIdx.y : blockIdx.y + 1) * blockDim.y + threadIdx.y;
    auto j = (blockIdx.x < p ? blockIdx.x : blockIdx.x + 1) * blockDim.x + threadIdx.x;
    auto ii = p * blockDim.y + threadIdx.y;
    auto jj = p * blockDim.x + threadIdx.x;
    auto blockSize = blockDim.x * blockDim.y;
    if (i < n && j < n) shared(threadIdx.y, threadIdx.x) = graph(i, j);
    else shared(threadIdx.y, threadIdx.x) = DATA_RANGE;
    if (i < n && jj < n) shared_offset(threadIdx.y, threadIdx.x, blockSize) = graph(i, jj);
    else shared_offset(threadIdx.y, threadIdx.x, blockSize) = DATA_RANGE;
    if (ii < n && j < n) shared_offset(threadIdx.y, threadIdx.x, blockSize * 2) = graph(ii, j);
    else shared_offset(threadIdx.y, threadIdx.x, blockSize * 2) = DATA_RANGE;
    __syncthreads();
    # pragma unroll 32
    for (int k = 0; k < BLOCK_SIZE; k++) {
        shared(threadIdx.y, threadIdx.x) = min(shared(threadIdx.y, threadIdx.x), shared_offset(threadIdx.y, k, blockSize) + shared_offset(k, threadIdx.x, blockSize * 2));
    }
    if (i < n && j < n) graph(i, j) = shared(threadIdx.y, threadIdx.x);
}

}

void apsp(int n, /* device */ int *graph) {
    const int b = 32;
    const int m = (n - 1) / b + 1;
    const dim3 thr(b, b);
    const dim3 blk(m, m);
    for (int p = 0; p < m; p++) {
        stage1<<<1, thr, b * b * sizeof(int)>>>(n, p, graph);
        stage2<<<dim3(m - 1, 2), thr, 2 * b * b * sizeof(int)>>>(n, p, graph);
        stage3<<<dim3(m - 1, m - 1), thr, 3 * b * b * sizeof(int)>>>(n, p, graph);
    }
}

