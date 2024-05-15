// PLEASE MODIFY THIS FILE TO IMPLEMENT YOUR SOLUTION

#include "apsp.h"

namespace APSP {

#define BLOCK_DIM 32
#define OFFSET 1024
#define DATA_RANGE 100000
#define graph(i, j) graph[(i) * n + (j)]
#define shared(i, j) shared[(i) * BLOCK_DIM + (j)]
#define shared0(i, j) shared0[(i) * BLOCK_DIM + (j)]
#define shared1(i, j) shared1[(i) * BLOCK_DIM + (j)]

__global__ void stage1(const int n, const int p, int *graph) {
    __shared__ int shared[OFFSET];
    auto i = p * BLOCK_DIM + threadIdx.y;
    auto j = p * BLOCK_DIM + threadIdx.x;
    if (i < n && j < n) shared(threadIdx.y, threadIdx.x) = graph(i, j);
    else shared(threadIdx.y, threadIdx.x) = DATA_RANGE;
    __syncthreads();
    int sum, tmp = shared(threadIdx.y, threadIdx.x);
    int* shared0 = shared + threadIdx.y * BLOCK_DIM;
    int* shared1 = shared + threadIdx.x;
    for (int k = 0; k < BLOCK_DIM; k++) {
        // tmp = min(tmp, shared(threadIdx.y, k) + shared(k, threadIdx.x));
        sum = *shared0 + *shared1;
        if (tmp > sum) tmp = sum;
        shared0++;
        shared1 += BLOCK_DIM;
    }
    if (i < n && j < n) graph(i, j) = tmp;
}

__global__ void stage2(const int n, const int p, int *graph) {
    __shared__ int shared[2 * OFFSET];
    if (blockIdx.y) {
        auto i = p * BLOCK_DIM + threadIdx.y;
        auto j = (blockIdx.x < p ? blockIdx.x : blockIdx.x + 1) * BLOCK_DIM + threadIdx.x;
        auto jj = p * BLOCK_DIM + threadIdx.x;
        int* shared0 = shared;
        if (i < n && j < n) shared0(threadIdx.y, threadIdx.x) = graph(i, j);
        else shared0(threadIdx.y, threadIdx.x) = DATA_RANGE;
        int* shared1 = shared + OFFSET;
        if (i < n && jj < n) shared1(threadIdx.y, threadIdx.x) = graph(i, jj);
        else shared1(threadIdx.y, threadIdx.x) = DATA_RANGE;
        __syncthreads();
        int sum, tmp = shared0(threadIdx.y, threadIdx.x);
        shared0 += threadIdx.x;
        shared1 += threadIdx.y * BLOCK_DIM;
        for (int k = 0; k < BLOCK_DIM; k++) {
            // tmp = min(tmp, shared1(threadIdx.y, k) + shared(k, threadIdx.x));
            sum = *shared1 + *shared0;
            if (tmp > sum) tmp = sum;
            shared0 += BLOCK_DIM;
            shared1++;
        }
        if (i < n && j < n) graph(i, j) = tmp;
    } else {
        auto i = (blockIdx.x < p ? blockIdx.x : blockIdx.x + 1) * BLOCK_DIM + threadIdx.y;
        auto j = p * BLOCK_DIM + threadIdx.x;
        auto ii = p * BLOCK_DIM + threadIdx.y;
        int* shared0 = shared;
        if (i < n && j < n) shared0(threadIdx.y, threadIdx.x) = graph(i, j);
        else shared0(threadIdx.y, threadIdx.x) = DATA_RANGE;
        int* shared1 = shared + OFFSET;
        if (ii < n && j < n) shared1(threadIdx.y, threadIdx.x) = graph(ii, j);
        else shared1(threadIdx.y, threadIdx.x) = DATA_RANGE;
        __syncthreads();
        int sum, tmp = shared0(threadIdx.y, threadIdx.x);
        shared0 += threadIdx.y * BLOCK_DIM;
        shared1 += threadIdx.x;
        for (int k = 0; k < BLOCK_DIM; k++) {
            // tmp = min(tmp, shared(threadIdx.y, k) + shared1(k, threadIdx.x));
            sum = *shared0 + *shared1;
            if (tmp > sum) tmp = sum;
            shared0++;
            shared1 += BLOCK_DIM;
        }
        if (i < n && j < n) graph(i, j) = tmp;
    }
}

__global__ void stage3_1(const int n, const int p, int *graph) {
    __shared__ int shared[2 * OFFSET];
    int* shared0 = shared;
    int* shared1 = shared + OFFSET;
    auto ii = p * BLOCK_DIM + threadIdx.y;
    auto jj = p * BLOCK_DIM + threadIdx.x;
    for (int m = 0; m < 1; m++) {
        auto i = blockIdx.y + m;
        if (i >= p) i++;
        i = i * BLOCK_DIM + threadIdx.y;
        auto j = blockIdx.x + m;
        if (j >= p) j++;
        j = j * BLOCK_DIM + threadIdx.x;
        if (i < n && jj < n) shared0(threadIdx.y, threadIdx.x) = graph(i, jj);
        else shared0(threadIdx.y, threadIdx.x) = DATA_RANGE;
        if (ii < n && j < n) shared1(threadIdx.y, threadIdx.x) = graph(ii, j);
        else shared1(threadIdx.y, threadIdx.x) = DATA_RANGE;
        shared0 += OFFSET;
        shared1 += OFFSET;
    }
    __syncthreads();
    shared0 = shared;
    for (int m = 0; m < 1; m++) {
        auto i = blockIdx.y + m;
        if (i >= p) i++;
        i = i * BLOCK_DIM + threadIdx.y;
        shared1 = shared + OFFSET;
        for (int l = 0; l < 1; l++) {
            auto j = blockIdx.x + l;
            if (j >= p) j++;
            j = j * BLOCK_DIM + threadIdx.x;
            int tmp, sum;
            if (i < n && j < n) tmp = graph(i, j);
            else tmp = DATA_RANGE;
            int* shared00 = shared0 + threadIdx.y * BLOCK_DIM;
            int* shared11 = shared1 + threadIdx.x;
            for (int k = 0; k < BLOCK_DIM; k++) {
                // tmp = min(tmp, shared0(threadIdx.y, k) + shared1(k, threadIdx.x));
                sum = *shared00 + *shared11;
                if (tmp > sum) tmp = sum;
                shared00++;
                shared11 += BLOCK_DIM;
            }
            if (i < n && j < n) graph(i, j) = tmp;
            shared1 += OFFSET;
        }
        shared0 += OFFSET;
    }
}

__global__ void stage3_2(const int n, const int p, int *graph) {
    __shared__ int shared[4 * OFFSET];
    int* shared0 = shared;
    int* shared1 = shared + 2 * OFFSET;
    auto ii = p * BLOCK_DIM + threadIdx.y;
    auto jj = p * BLOCK_DIM + threadIdx.x;
    for (int m = 0; m < 2; m++) {
        auto i = 2 * blockIdx.y + m;
        if (i >= p) i++;
        i = i * BLOCK_DIM + threadIdx.y;
        auto j = 2 * blockIdx.x + m;
        if (j >= p) j++;
        j = j * BLOCK_DIM + threadIdx.x;
        if (i < n && jj < n) shared0(threadIdx.y, threadIdx.x) = graph(i, jj);
        else shared0(threadIdx.y, threadIdx.x) = DATA_RANGE;
        if (ii < n && j < n) shared1(threadIdx.y, threadIdx.x) = graph(ii, j);
        else shared1(threadIdx.y, threadIdx.x) = DATA_RANGE;
        shared0 += OFFSET;
        shared1 += OFFSET;
    }
    __syncthreads();
    shared0 = shared;
    for (int m = 0; m < 2; m++) {
        auto i = 2 * blockIdx.y + m;
        if (i >= p) i++;
        i = i * BLOCK_DIM + threadIdx.y;
        shared1 = shared + 2 * OFFSET;
        for (int l = 0; l < 2; l++) {
            auto j = 2 * blockIdx.x + l;
            if (j >= p) j++;
            j = j * BLOCK_DIM + threadIdx.x;
            int tmp, sum;
            if (i < n && j < n) tmp = graph(i, j);
            else tmp = DATA_RANGE;
            int* shared00 = shared0 + threadIdx.y * BLOCK_DIM;
            int* shared11 = shared1 + threadIdx.x;
            for (int k = 0; k < BLOCK_DIM; k++) {
                // tmp = min(tmp, shared0(threadIdx.y, k) + shared1(k, threadIdx.x));
                sum = *shared00 + *shared11;
                if (tmp > sum) tmp = sum;
                shared00++;
                shared11 += BLOCK_DIM;
            }
            if (i < n && j < n) graph(i, j) = tmp;
            shared1 += OFFSET;
        }
        shared0 += OFFSET;
    }
}

__global__ void stage3_4(const int n, const int p, int *graph) {
    __shared__ int shared[8 * OFFSET];
    int* shared0 = shared;
    int* shared1 = shared + 4 * OFFSET;
    auto ii = p * BLOCK_DIM + threadIdx.y;
    auto jj = p * BLOCK_DIM + threadIdx.x;
    for (int m = 0; m < 4; m++) {
        auto i = 4 * blockIdx.y + m;
        if (i >= p) i++;
        i = i * BLOCK_DIM + threadIdx.y;
        auto j = 4 * blockIdx.x + m;
        if (j >= p) j++;
        j = j * BLOCK_DIM + threadIdx.x;
        if (i < n && jj < n) shared0(threadIdx.y, threadIdx.x) = graph(i, jj);
        else shared0(threadIdx.y, threadIdx.x) = DATA_RANGE;
        if (ii < n && j < n) shared1(threadIdx.y, threadIdx.x) = graph(ii, j);
        else shared1(threadIdx.y, threadIdx.x) = DATA_RANGE;
        shared0 += OFFSET;
        shared1 += OFFSET;
    }
    __syncthreads();
    shared0 = shared;
    for (int m = 0; m < 4; m++) {
        auto i = 4 * blockIdx.y + m;
        if (i >= p) i++;
        i = i * BLOCK_DIM + threadIdx.y;
        shared1 = shared + 4 * OFFSET;
        for (int l = 0; l < 4; l++) {
            auto j = 4 * blockIdx.x + l;
            if (j >= p) j++;
            j = j * BLOCK_DIM + threadIdx.x;
            int tmp, sum;
            if (i < n && j < n) tmp = graph(i, j);
            else tmp = DATA_RANGE;
            int* shared00 = shared0 + threadIdx.y * BLOCK_DIM;
            int* shared11 = shared1 + threadIdx.x;
            for (int k = 0; k < BLOCK_DIM; k++) {
                // tmp = min(tmp, shared0(threadIdx.y, k) + shared1(k, threadIdx.x));
                sum = *shared00 + *shared11;
                if (tmp > sum) tmp = sum;
                shared00++;
                shared11 += BLOCK_DIM;
            }
            if (i < n && j < n) graph(i, j) = tmp;
            shared1 += OFFSET;
        }
        shared0 += OFFSET;
    }
}

__global__ void stage3_6(const int n, const int p, int *graph) {
    __shared__ int shared[12 * OFFSET];
    int* shared0 = shared;
    int* shared1 = shared + 6 * OFFSET;
    auto ii = p * BLOCK_DIM + threadIdx.y;
    auto jj = p * BLOCK_DIM + threadIdx.x;
    for (int m = 0; m < 6; m++) {
        auto i = 6 * blockIdx.y + m;
        if (i >= p) i++;
        i = i * BLOCK_DIM + threadIdx.y;
        auto j = 6 * blockIdx.x + m;
        if (j >= p) j++;
        j = j * BLOCK_DIM + threadIdx.x;
        if (i < n && jj < n) shared0(threadIdx.y, threadIdx.x) = graph(i, jj);
        else shared0(threadIdx.y, threadIdx.x) = DATA_RANGE;
        if (ii < n && j < n) shared1(threadIdx.y, threadIdx.x) = graph(ii, j);
        else shared1(threadIdx.y, threadIdx.x) = DATA_RANGE;
        shared0 += OFFSET;
        shared1 += OFFSET;
    }
    __syncthreads();
    shared0 = shared;
    for (int m = 0; m < 6; m++) {
        auto i = 6 * blockIdx.y + m;
        if (i >= p) i++;
        i = i * BLOCK_DIM + threadIdx.y;
        shared1 = shared + 6 * OFFSET;
        for (int l = 0; l < 6; l++) {
            auto j = 6 * blockIdx.x + l;
            if (j >= p) j++;
            j = j * BLOCK_DIM + threadIdx.x;
            int tmp, sum;
            if (i < n && j < n) tmp = graph(i, j);
            else tmp = DATA_RANGE;
            int* shared00 = shared0 + threadIdx.y * BLOCK_DIM;
            int* shared11 = shared1 + threadIdx.x;
            for (int k = 0; k < BLOCK_DIM; k++) {
                // tmp = min(tmp, shared0(threadIdx.y, k) + shared1(k, threadIdx.x));
                sum = *shared00 + *shared11;
                if (tmp > sum) tmp = sum;
                shared00++;
                shared11 += BLOCK_DIM;
            }
            if (i < n && j < n) graph(i, j) = tmp;
            shared1 += OFFSET;
        }
        shared0 += OFFSET;
    }
}

}

void apsp(const int n, /* device */ int *graph) {
    constexpr int b = 32;
    constexpr dim3 thr(b, b);
    const int m = (n - 1) / b + 1;
    const dim3 blk2(m - 1, 2);
    if (n <= 32) {
        APSP::stage1<<<1, thr>>>(n, 0, graph);
    } else if (n < 700) {
        const int batch = 1;
        const int dim = (m - 2) / batch + 1;
        const dim3 blk3(dim, dim);
        for (int p = 0; p < m; p++) {
            APSP::stage1<<<1, thr>>>(n, p, graph);
            APSP::stage2<<<blk2, thr>>>(n, p, graph);
            APSP::stage3_1<<<blk3, thr>>>(n, p, graph);
        }
    } else if (n < 1050) {
        const int batch = 2;
        const int dim = (m - 2) / batch + 1;
        const dim3 blk3(dim, dim);
        for (int p = 0; p < m; p++) {
            APSP::stage1<<<1, thr>>>(n, p, graph);
            APSP::stage2<<<blk2, thr>>>(n, p, graph);
            APSP::stage3_2<<<blk3, thr>>>(n, p, graph);
        }
    } else if (n < 3050) {
        const int batch = 4;
        const int dim = (m - 2) / batch + 1;
        const dim3 blk3(dim, dim);
        for (int p = 0; p < m; p++) {
            APSP::stage1<<<1, thr>>>(n, p, graph);
            APSP::stage2<<<blk2, thr>>>(n, p, graph);
            APSP::stage3_4<<<blk3, thr>>>(n, p, graph);
        }
    } else {
        const int batch = 6;
        const int dim = (m - 2) / batch + 1;
        const dim3 blk3(dim, dim);
        for (int p = 0; p < m; p++) {
            APSP::stage1<<<1, thr>>>(n, p, graph);
            APSP::stage2<<<blk2, thr>>>(n, p, graph);
            APSP::stage3_6<<<blk3, thr>>>(n, p, graph);
        }
    }
}

