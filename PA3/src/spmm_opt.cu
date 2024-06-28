#include "spmm_opt.h"
#include "util.h"
#include "cuda_profiler_api.h"

#define ROW_NUM_32 1
#define ROW_THREAD_32 32
#define ROW_ELEM_32 64
#define ROW_NUM_256 1
#define ROW_THREAD_256 128
#define ROW_ELEM_256 128

__global__ void spmm_kernel_opt32(int *ptr, int *idx, float *val, float *vin, float *vout)
{
    __shared__ int s_idx[ROW_NUM_32 * ROW_ELEM_32];
    __shared__ float s_val[ROW_NUM_32 * ROW_ELEM_32];
    // int tid = blockIdx.x * blockDim.y + threadIdx.y;
    // if (tid >= num_v) return;
    int begin = ptr[blockIdx.x];
    int len = ptr[blockIdx.x + 1] - begin;
    int offset = len - threadIdx.x;
    // int offset = threadIdx.y * ROW_ELEM_32;
    // int *s_idx_base = s_idx + offset;
    // float *s_val_base = s_val + offset;
    int *s_idx_base = s_idx + threadIdx.x;
    float *s_val_base = s_val + threadIdx.x;
    float *vin_base = vin + threadIdx.x;
    int *idx_base = idx + begin + threadIdx.x;
    float *val_base = val + begin + threadIdx.x;

    float tmp = 0;
    for (int i = 0; i < len; i += ROW_ELEM_32)
    {
        // Load data into shared memory
        if (i < offset)
        {
            *s_idx_base = idx_base[i];
            *s_val_base = val_base[i];
        }
        int ii = i + ROW_THREAD_32;
        if (ii < offset)
        {
            s_idx_base[ROW_THREAD_32] = idx_base[ii];
            s_val_base[ROW_THREAD_32] = val_base[ii];
        }
        __syncwarp();

        // Compute
        int max = min(ROW_ELEM_32, len - i);
        for (int j = 0; j < max; j++)
        {
            tmp += vin_base[(s_idx[j] << 5)] * s_val[j];
        }
        __syncwarp();
    }
    // vout[(tid << 5) + threadIdx.x] = tmp;
    vout[(blockIdx.x << 5) + threadIdx.x] = tmp;
}

__global__ void spmm_kernel_opt256(int *ptr, int *idx, float *val, float *vin, float *vout)
{
    __shared__ int s_idx[ROW_NUM_256 * ROW_ELEM_256];
    __shared__ float s_val[ROW_NUM_256 * ROW_ELEM_256];
    // int tid = blockIdx.x * blockDim.y + threadIdx.y;
    // if (tid >= num_v) return;
    // int begin = ptr[tid], end = ptr[tid + 1];
    int begin = ptr[blockIdx.x], end = ptr[blockIdx.x + 1];
    // int offset = threadIdx.y * ROW_ELEM_256;
    // int *s_idx_base = s_idx + offset;
    // float *s_val_base = s_val + offset;
    int *s_idx_base = s_idx + threadIdx.x;
    float *s_val_base = s_val + threadIdx.x;
    float *vin_base1 = vin + threadIdx.x;
    float *vin_base2 = vin + threadIdx.x + ROW_THREAD_256;

    float tmp1 = 0, tmp2 = 0;
    for (int i = begin; i < end; i += ROW_ELEM_256)
    {
        // Load data into shared memory
        int ii = i + threadIdx.x;
        if (ii < end)
        {
            *s_idx_base = idx[ii];
            *s_val_base = val[ii];
        }
        __syncthreads();

        // Compute
        int max = min(ROW_ELEM_256, end - i);
        for (int j = 0; j < max; ++j)
        {
            int tmp_idx = s_idx[j] << 8;
            float tmp_val = s_val[j];
            tmp1 += vin_base1[tmp_idx] * tmp_val;
            tmp2 += vin_base2[tmp_idx] * tmp_val;
        }
        __syncthreads();
    }
    // vout[(tid << 8) + threadIdx.x] = tmp1;
    // vout[(tid << 8) + threadIdx.x + ROW_THREAD_256] = tmp2;
    vout[(blockIdx.x << 8) + threadIdx.x] = tmp1;
    vout[(blockIdx.x << 8) + threadIdx.x + ROW_THREAD_256] = tmp2;
}

void SpMMOpt::preprocess(float *vin, float *vout)
{
    if (feat_in == 32)
    {
        block.y = ROW_NUM_32;
        // grid.x = (num_v + block.y - 1) / block.y;
        grid.x = num_v;
        block.x = ROW_THREAD_32;
    }
    else
    {
        block.y = ROW_NUM_256;
        // grid.x = (num_v + block.y - 1) / block.y;
        grid.x = num_v;
        block.x = ROW_THREAD_256;
    }
}

void SpMMOpt::run(float *vin, float *vout)
{
    checkCudaErrors(cudaProfilerStart());
    if (feat_in == 32)
    {
        spmm_kernel_opt32<<<grid, block>>>(d_ptr, d_idx, d_val, vin, vout);
    }
    else
    {
        spmm_kernel_opt256<<<grid, block>>>(d_ptr, d_idx, d_val, vin, vout);
    }
    checkCudaErrors(cudaProfilerStop());
}
