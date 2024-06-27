#include "spmm_opt.h"

__global__ void spmm_kernel_opt32(int *ptr, int *idx, float *val, float *vin, float *vout, int num_v)
{
    __shared__ int s_idx[32 * 32];
    __shared__ float s_val[32 * 32];
    int tid = blockIdx.x * blockDim.y + threadIdx.y;
    if (tid >= num_v) return;
    int begin = ptr[tid], end = ptr[tid + 1];
    int tmp = 0;
    for (int i = begin; i < end; i += 32)
    {
        // Load data into shared memory
        int ii = i + threadIdx.x;
        int offset = threadIdx.y * 32 + threadIdx.x;
        if (ii < end)
        {
            s_idx[offset] = idx[ii];
            s_val[offset] = val[ii];
        }
        __syncwarp();

        // Compute
        int max = threadIdx.y * 32 + min(32, end - i);
        for (int j = threadIdx.y * 32; j < max; ++j)
        {
            tmp += vin[s_idx[j] * 32 + threadIdx.x] * s_val[j];
        }
        __syncwarp();
    }
    vout[tid * 32 + threadIdx.x] = tmp;
}

__global__ void spmm_kernel_opt256(int *ptr, int *idx, float *val, float *vin, float *vout, int num_v)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_v) return;
    int begin = ptr[tid], end = ptr[tid + 1];
    for (int j = 0; j < 256; ++j)
    {
        float result = 0.0f;
        for (int i = begin; i < end; ++i)
        {
            result += vin[idx[i] * 256 + j] * val[i];
        }
        vout[tid * 256 + j] = result;
    }
}

void SpMMOpt::preprocess(float *vin, float *vout)
{
    if (feat_in == 32)
    {
        block.y = 32;
        grid.x = (num_v + block.y - 1) / block.y;
        block.x = 32;
    }
    else
    {
        block.y = 8;
        grid.x = (num_v + block.y - 1) / block.y;
        block.x = 128;
    }
}

void SpMMOpt::run(float *vin, float *vout)
{
    if (feat_in == 32)
    {
        spmm_kernel_opt32<<<grid, block>>>(d_ptr, d_idx, d_val, vin, vout, num_v);
    }
    else
    {
        spmm_kernel_opt256<<<grid, block>>>(d_ptr, d_idx, d_val, vin, vout, num_v);
    }
}
