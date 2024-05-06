// PLEASE MODIFY THIS FILE TO IMPLEMENT YOUR SOLUTION

// Brute Force APSP Implementation:

#include "cuda_utils.h"
#include "apsp.h"

namespace {

#define index(i, j, n) ((i) * (n) + (j))

__global__ void stage1(int n, int p, int *graph) {
    extern __shared__ int shared[];
    auto i = p * blockDim.y + threadIdx.y;
    auto j = p * blockDim.x + threadIdx.x;
    bool in_range = i < n && j < n;
    if (in_range) {
        shared[index(threadIdx.y, threadIdx.x, blockDim.x)] = graph[index(i, j, n)];
        # pragma unroll 32
        for (int k = 0; k < blockDim.x; k++) {
            __syncthreads();
            shared[index(threadIdx.y, threadIdx.x, blockDim.x)] = min(shared[index(threadIdx.y, threadIdx.x, blockDim.x)], shared[index(threadIdx.y, k, blockDim.x)] + shared[index(k, threadIdx.x, blockDim.x)]);
        }
        graph[index(i, j, n)] = shared[index(threadIdx.y, threadIdx.x, blockDim.x)];
    }
}

__global__ void stage2(int n, int p, int *graph) {
    extern __shared__ int shared[];
    auto i = (blockIdx.y ? p : (blockIdx.x < p ? blockIdx.x : blockIdx.x + 1)) * blockDim.y + threadIdx.y;
    auto j = (blockIdx.y ? (blockIdx.x < p ? blockIdx.x : blockIdx.x + 1) : p) * blockDim.x + threadIdx.x;
    bool in_range = i < n && j < n;
    if (in_range) {
        shared[index(threadIdx.y, threadIdx.x, blockDim.x)] = graph[index(i, j, n)];
        auto ii = p * blockDim.y + threadIdx.y;
        auto jj = p * blockDim.x + threadIdx.x;
        auto blockSize = blockDim.x * blockDim.y;
        shared[blockSize + index(threadIdx.y, threadIdx.x, blockDim.x)] = graph[index(ii, jj, n)];
        # pragma unroll 32
        for (int k = 0; k < blockDim.x; k++) {
            __syncthreads();
            shared[index(threadIdx.y, threadIdx.x, blockDim.x)] = min(shared[index(threadIdx.y, threadIdx.x, blockDim.x)], shared[(blockIdx.y ? blockSize : 0) + index(threadIdx.y, k, blockDim.x)] + shared[(blockIdx.y ? 0 : blockSize) + index(k, threadIdx.x, blockDim.x)]);
        }
        graph[index(i, j, n)] = shared[index(threadIdx.y, threadIdx.x, blockDim.x)];
    }
}

__global__ void stage3(int n, int p, int *graph) {
    extern __shared__ int shared[];
    auto i = (blockIdx.y < p ? blockIdx.y : blockIdx.y + 1) * blockDim.y + threadIdx.y;
    auto j = (blockIdx.x < p ? blockIdx.x : blockIdx.x + 1) * blockDim.x + threadIdx.x;
    bool in_range = i < n && j < n;
    if (in_range) {
        shared[index(threadIdx.y, threadIdx.x, blockDim.x)] = graph[index(i, j, n)];
        auto ii = p * blockDim.y + threadIdx.y;
        auto jj = p * blockDim.x + threadIdx.x;
        auto blockSize = blockDim.x * blockDim.y;
        shared[blockSize + index(threadIdx.y, threadIdx.x, blockDim.x)] = graph[index(i, jj, n)];
        shared[2 * blockSize + index(threadIdx.y, threadIdx.x, blockDim.x)] = graph[index(ii, j, n)];
        # pragma unroll 32
        for (int k = 0; k < blockDim.x; k++) {
            __syncthreads();
            shared[index(threadIdx.y, threadIdx.x, blockDim.x)] = min(shared[index(threadIdx.y, threadIdx.x, blockDim.x)], shared[blockSize + index(threadIdx.y, k, blockDim.x)] + shared[2 * blockSize + index(k, threadIdx.x, blockDim.x)]);
        }
        graph[index(i, j, n)] = shared[index(threadIdx.y, threadIdx.x, blockDim.x)];
    }
}

}

void apsp(int n, /* device */ int *graph) {
    const int b = 32;
    const int m = (n - 1) / b + 1;
    const dim3 thr(b, b);
    const dim3 blk(m, m);
    for (int p = 0; p < m; p++) {
        CHK_CUDA_ERR(stage1<<<1, thr, b * b * sizeof(int)>>>(n, p, graph));
        CHK_CUDA_ERR(stage2<<<dim3(m - 1, 2), thr, 2 * b * b * sizeof(int)>>>(n, p, graph));
        CHK_CUDA_ERR(stage3<<<dim3(m - 1, m - 1), thr, 3 * b * b * sizeof(int)>>>(n, p, graph));
    }
}

