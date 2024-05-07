// PLEASE MODIFY THIS FILE TO IMPLEMENT YOUR SOLUTION

#include "apsp.h"

namespace APSP {

#define BLOCK_SIZE 32
#define DATA_RANGE 100000
#define graph(i, j) graph[(i) * n + (j)]
#define shared(i, j) shared[(i) * blockDim.x + (j)]
#define shared_offset(i, j, offset) shared[(i) * blockDim.x + (j) + offset]

__global__ void stage1(int n, int p, int *graph) {
    __shared__ int shared[BLOCK_SIZE * BLOCK_SIZE];
    auto i = p * blockDim.y + threadIdx.y;
    auto j = p * blockDim.x + threadIdx.x;
    if (i < n && j < n) shared(threadIdx.y, threadIdx.x) = graph(i, j);
    else shared(threadIdx.y, threadIdx.x) = DATA_RANGE;
    __syncthreads();
    int tmp = shared(threadIdx.y, threadIdx.x);
    # pragma unroll 32
    for (int k = 0; k < BLOCK_SIZE; k++) {
        tmp = min(tmp, shared(threadIdx.y, k) + shared(k, threadIdx.x));
    }
    if (i < n && j < n) graph(i, j) = tmp;
}

__global__ void stage2(int n, int p, int *graph) {
    __shared__ int shared[2 * BLOCK_SIZE * BLOCK_SIZE];
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
    int tmp = shared(threadIdx.y, threadIdx.x);
    if (blockIdx.y) {
        # pragma unroll 32
        for (int k = 0; k < BLOCK_SIZE; k++) {
            tmp = min(tmp, shared_offset(threadIdx.y, k, blockSize) + shared(k, threadIdx.x));
        }
    } else {
        # pragma unroll 32
        for (int k = 0; k < BLOCK_SIZE; k++) {
            tmp = min(tmp, shared(threadIdx.y, k) + shared_offset(k, threadIdx.x, blockSize));
        }
    }
    if (i < n && j < n) graph(i, j) = tmp;
}

__global__ void stage3(int n, int p, int *graph) {
    __shared__ int shared[2 * BLOCK_SIZE * BLOCK_SIZE];
    auto i = (blockIdx.y < p ? blockIdx.y : blockIdx.y + 1) * blockDim.y + threadIdx.y;
    auto j = (blockIdx.x < p ? blockIdx.x : blockIdx.x + 1) * blockDim.x + threadIdx.x;
    auto ii = p * blockDim.y + threadIdx.y;
    auto jj = p * blockDim.x + threadIdx.x;
    auto blockSize = blockDim.x * blockDim.y;
    if (i < n && jj < n) shared(threadIdx.y, threadIdx.x) = graph(i, jj);
    else shared(threadIdx.y, threadIdx.x) = DATA_RANGE;
    if (ii < n && j < n) shared_offset(threadIdx.y, threadIdx.x, blockSize) = graph(ii, j);
    else shared_offset(threadIdx.y, threadIdx.x, blockSize) = DATA_RANGE;
    __syncthreads();
    int tmp;
    if (i < n && j < n) tmp = graph(i, j);
    else tmp = DATA_RANGE;
    # pragma unroll 32
    for (int k = 0; k < BLOCK_SIZE; k++) {
        tmp = min(tmp, shared(threadIdx.y, k) + shared_offset(k, threadIdx.x, blockSize));
    }
    if (i < n && j < n) graph(i, j) = tmp;
}

}

void apsp(int n, /* device */ int *graph) {
    const int b = 32;
    const int m = (n - 1) / b + 1;
    const dim3 thr(b, b);
    const dim3 blk(m, m);
    for (int p = 0; p < m; p++) {
        APSP::stage1<<<1, thr>>>(n, p, graph);
        APSP::stage2<<<dim3(m - 1, 2), thr>>>(n, p, graph);
        APSP::stage3<<<dim3(m - 1, m - 1), thr>>>(n, p, graph);
    }
}

