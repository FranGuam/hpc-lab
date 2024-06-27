#include "spmm_opt.h"
#include "util.h"
#include "cuda_profiler_api.h"

__global__ void spmm_kernel_opt32(int *ptr, int *idx, float *val, float *vin, float *vout, int num_v)
{
    __shared__ int s_idx[1 * 128];
    __shared__ float s_val[1 * 128];
    int offset = threadIdx.y << 5; // threadIdx.y * 32
    int *s_idx_base = s_idx + offset;
    float *s_val_base = s_val + offset;
    int tid = blockIdx.x * blockDim.y + threadIdx.y;
    if (tid >= num_v) return;
    int begin = ptr[tid], end = ptr[tid + 1];
    float tmp = 0;
    for (int i = begin; i < end; i += 128)
    {
        // Load data into shared memory
        int ii = i + threadIdx.x;
        if (ii < end)
        {
            s_idx_base[threadIdx.x] = idx[ii];
            s_val_base[threadIdx.x] = val[ii];
        }
        if (ii + 32 < end)
        {
            s_idx_base[threadIdx.x + 32] = idx[ii + 32];
            s_val_base[threadIdx.x + 32] = val[ii + 32];
        }
        if (ii + 64 < end)
        {
            s_idx_base[threadIdx.x + 64] = idx[ii + 64];
            s_val_base[threadIdx.x + 64] = val[ii + 64];
        }
        if (ii + 96 < end)
        {
            s_idx_base[threadIdx.x + 96] = idx[ii + 96];
            s_val_base[threadIdx.x + 96] = val[ii + 96];
        }
        __syncwarp();

        // Compute
        int max = min(128, end - i);
        for (int j = 0; j < max; ++j)
        {
            tmp += vin[(s_idx_base[j] << 5) + threadIdx.x] * s_val_base[j];
        }
        __syncwarp();
    }
    vout[(tid << 5) + threadIdx.x] = tmp;
}

__global__ void spmm_kernel_opt256(int *ptr, int *idx, float *val, float *vin, float *vout, int num_v)
{
    __shared__ int s_idx[1 * 256];
    __shared__ float s_val[1 * 256];
    int offset = threadIdx.y << 8; // threadIdx.y * 256
    int *s_idx_base = s_idx + offset;
    float *s_val_base = s_val + offset;
    int tid = blockIdx.x * blockDim.y + threadIdx.y;
    if (tid >= num_v) return;
    int begin = ptr[tid], end = ptr[tid + 1];
    float tmp1 = 0, tmp2 = 0;
    for (int i = begin; i < end; i += 128)
    {
        // Load data into shared memory
        int ii = i + threadIdx.x;
        if (ii < end)
        {
            s_idx_base[threadIdx.x] = idx[ii];
            s_val_base[threadIdx.x] = val[ii];
        }
        __syncthreads();

        // Compute
        int max = min(128, end - i);
        for (int j = 0; j < max; ++j)
        {
            tmp1 += vin[(s_idx_base[j] << 8) + threadIdx.x] * s_val_base[j];
            tmp2 += vin[(s_idx_base[j] << 8) + threadIdx.x + 128] * s_val_base[j];
        }
        __syncthreads();
    }
    vout[(tid << 8) + threadIdx.x] = tmp1;
    vout[(tid << 8) + threadIdx.x + 128] = tmp2;
}

void SpMMOpt::preprocess(float *vin, float *vout)
{
    if (feat_in == 32)
    {
        block.y = 1;
        grid.x = (num_v + block.y - 1) / block.y;
        block.x = 32;
    }
    else
    {
        block.y = 1;
        grid.x = (num_v + block.y - 1) / block.y;
        block.x = 128;
    }
}

void SpMMOpt::run(float *vin, float *vout)
{
    checkCudaErrors(cudaProfilerStart());
    if (feat_in == 32)
    {
        spmm_kernel_opt32<<<grid, block>>>(d_ptr, d_idx, d_val, vin, vout, num_v);
    }
    else
    {
        spmm_kernel_opt256<<<grid, block>>>(d_ptr, d_idx, d_val, vin, vout, num_v);
    }
    checkCudaErrors(cudaProfilerStop());
}
