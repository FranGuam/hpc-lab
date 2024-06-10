#include "spmm_opt.h"
#include "util.h"

__global__ void csr2coo_kernel(int *ptr, int *coo, int num_v)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_v) return;
    int begin = ptr[tid], end = ptr[tid + 1];
    for (int i = begin; i < end; ++i)
    {
        coo[i] = tid;
    }
}

__global__ void column_major_kernel(int *coo, int *idx, float *val, int *col_idx, int *row_idx, float *value, int num_e)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_e) return;
    col_idx[tid] = idx[tid];
    row_idx[tid] = coo[tid];
    value[tid] = val[tid];
}

__global__ void spmm_kernel_opt(int *ptr, int *idx, float *val, float *vin, float *vout, int num_v, int INFEATURE, int *coo, int num_e)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.y;
    if (tid >= num_e) return;
    int row = coo[tid];
    int col = idx[tid];
    float value = val[tid];
    int src_idx = col * INFEATURE + threadIdx.x;
    int dst_idx = row * INFEATURE + threadIdx.x;
    if (INFEATURE == 32) {
        atomicAdd(&vout[dst_idx], vin[src_idx] * value);
    }
    else 
    {
        for (int j = 0; j < 256; j += 32)
        {
            atomicAdd(&vout[dst_idx + j], vin[src_idx + j] * value);
        }
    }
}

void SpMMOpt::preprocess(float *vin, float *vout)
{
    int block_size, num_blocks;
    // Convert CSR to COO
    block_size = 128;
    num_blocks = (num_v + block_size - 1) / block_size;
    checkCudaErrors(cudaMalloc2((void**)&d_coo, num_e * sizeof(int)));
    csr2coo_kernel<<<num_blocks, block_size>>>(d_ptr, d_coo, num_v);

    // Convert Row-major to Column-major
    block_size = 128;
    num_blocks = (num_v + block_size - 1) / block_size;
    checkCudaErrors(cudaMalloc2((void**)&d_col_idx, num_e * sizeof(int)));
    checkCudaErrors(cudaMalloc2((void**)&d_row_idx, num_e * sizeof(int)));
    checkCudaErrors(cudaMalloc2((void**)&d_value, num_e * sizeof(float)));
    column_major_kernel<<<num_blocks, block_size>>>(d_coo, d_idx, d_val, d_col_idx, d_row_idx, d_value, num_e);

    // Decide grid and block size for spmm_kernel_opt
    int BLOCK_SIZE = 32;
    grid.x = (num_e + BLOCK_SIZE - 1) / BLOCK_SIZE;
    block.y = BLOCK_SIZE;
    block.x = 32;
}

void SpMMOpt::run(float *vin, float *vout)
{
    // TODO: your code
    spmm_kernel_opt<<<grid, block>>>(d_ptr, d_idx, d_val, vin, vout, num_v, feat_in, d_coo, num_e);
}