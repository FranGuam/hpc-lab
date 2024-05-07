// PLEASE MODIFY THIS FILE TO IMPLEMENT YOUR SOLUTION

#include "apsp.h"

namespace APSP {

#define BLOCK_SIZE 32
#define DATA_RANGE 100000
#define graph(i, j) graph[(i) * n + (j)]
#define shared(i, j) shared[(i) * BLOCK_SIZE + (j)]
#define shared0(i, j) shared0[(i) * BLOCK_SIZE + (j)]
#define shared1(i, j) shared1[(i) * BLOCK_SIZE + (j)]

__global__ void stage1(int n, int p, int *graph) {
    __shared__ int shared[BLOCK_SIZE * BLOCK_SIZE];
    auto i = p * BLOCK_SIZE + threadIdx.y;
    auto j = p * BLOCK_SIZE + threadIdx.x;
    if (i < n && j < n) shared(threadIdx.y, threadIdx.x) = graph(i, j);
    else shared(threadIdx.y, threadIdx.x) = DATA_RANGE;
    __syncthreads();
    int sum, tmp = shared(threadIdx.y, threadIdx.x);
    int* shared0 = shared + threadIdx.y * BLOCK_SIZE;
    int* shared1 = shared + threadIdx.x;
    # pragma unroll 32
    for (int k = 0; k < BLOCK_SIZE; k++) {
        // tmp = min(tmp, shared(threadIdx.y, k) + shared(k, threadIdx.x));
        sum = *shared0 + *shared1;
        if (tmp > sum) tmp = sum;
        shared0++;
        shared1 += BLOCK_SIZE;
    }
    if (i < n && j < n) graph(i, j) = tmp;
}

__global__ void stage2(int n, int p, int *graph) {
    __shared__ int shared[2 * BLOCK_SIZE * BLOCK_SIZE];
    if (blockIdx.y) {
        auto i = p * BLOCK_SIZE + threadIdx.y;
        auto j = (blockIdx.x < p ? blockIdx.x : blockIdx.x + 1) * BLOCK_SIZE + threadIdx.x;
        auto jj = p * BLOCK_SIZE + threadIdx.x;
        int* shared0 = shared;
        if (i < n && j < n) shared0(threadIdx.y, threadIdx.x) = graph(i, j);
        else shared0(threadIdx.y, threadIdx.x) = DATA_RANGE;
        int* shared1 = shared + BLOCK_SIZE * BLOCK_SIZE;
        if (i < n && jj < n) shared1(threadIdx.y, threadIdx.x) = graph(i, jj);
        else shared1(threadIdx.y, threadIdx.x) = DATA_RANGE;
        __syncthreads();
        int sum, tmp = shared0(threadIdx.y, threadIdx.x);
        shared0 += threadIdx.x;
        shared1 += threadIdx.y * BLOCK_SIZE;
        # pragma unroll 32
        for (int k = 0; k < BLOCK_SIZE; k++) {
            // tmp = min(tmp, shared1(threadIdx.y, k) + shared(k, threadIdx.x));
            sum = *shared1 + *shared0;
            if (tmp > sum) tmp = sum;
            shared0 += BLOCK_SIZE;
            shared1++;
        }
        if (i < n && j < n) graph(i, j) = tmp;
    } else {
        auto i = (blockIdx.x < p ? blockIdx.x : blockIdx.x + 1) * BLOCK_SIZE + threadIdx.y;
        auto j = p * BLOCK_SIZE + threadIdx.x;
        auto ii = p * BLOCK_SIZE + threadIdx.y;
        int* shared0 = shared;
        if (i < n && j < n) shared0(threadIdx.y, threadIdx.x) = graph(i, j);
        else shared0(threadIdx.y, threadIdx.x) = DATA_RANGE;
        int* shared1 = shared + BLOCK_SIZE * BLOCK_SIZE;
        if (ii < n && j < n) shared1(threadIdx.y, threadIdx.x) = graph(ii, j);
        else shared1(threadIdx.y, threadIdx.x) = DATA_RANGE;
        __syncthreads();
        int sum, tmp = shared0(threadIdx.y, threadIdx.x);
        shared0 += threadIdx.y * BLOCK_SIZE;
        shared1 += threadIdx.x;
        # pragma unroll 32
        for (int k = 0; k < BLOCK_SIZE; k++) {
            // tmp = min(tmp, shared(threadIdx.y, k) + shared1(k, threadIdx.x));
            sum = *shared0 + *shared1;
            if (tmp > sum) tmp = sum;
            shared0++;
            shared1 += BLOCK_SIZE;
        }
        if (i < n && j < n) graph(i, j) = tmp;
    }
}

__global__ void stage3(int n, int p, int *graph) {
    __shared__ int shared[2 * BLOCK_SIZE * BLOCK_SIZE];
    auto i = (blockIdx.y < p ? blockIdx.y : blockIdx.y + 1) * BLOCK_SIZE + threadIdx.y;
    auto j = (blockIdx.x < p ? blockIdx.x : blockIdx.x + 1) * BLOCK_SIZE + threadIdx.x;
    auto ii = p * BLOCK_SIZE + threadIdx.y;
    auto jj = p * BLOCK_SIZE + threadIdx.x;
    int* shared0 = shared;
    if (i < n && jj < n) shared0(threadIdx.y, threadIdx.x) = graph(i, jj);
    else shared0(threadIdx.y, threadIdx.x) = DATA_RANGE;
    int* shared1 = shared + BLOCK_SIZE * BLOCK_SIZE;
    if (ii < n && j < n) shared1(threadIdx.y, threadIdx.x) = graph(ii, j);
    else shared1(threadIdx.y, threadIdx.x) = DATA_RANGE;
    __syncthreads();
    int tmp, sum;
    if (i < n && j < n) tmp = graph(i, j);
    else tmp = DATA_RANGE;
    shared0 += threadIdx.y * BLOCK_SIZE;
    shared1 += threadIdx.x;
    # pragma unroll 32
    for (int k = 0; k < BLOCK_SIZE; k++) {
        // tmp = min(tmp, shared(threadIdx.y, k) + shared1(k, threadIdx.x));
        sum = *shared0 + *shared1;
        if (tmp > sum) tmp = sum;
        shared0++;
        shared1 += BLOCK_SIZE;
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

