#include "spmm_opt.h"
#include "util.h"

#define SMALL_WIDTH 32
#define LARGE_WIDTH 256
#define BLOCK_LEN 32

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

__global__ void spmm_kernel_opt(int *col_idx, int *row_idx, float *value, float *vin, float *vout, int num_v, int num_e, int feat_in)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.y;
    if (tid >= num_e) return;
    int row = row_idx[tid];
    int col = col_idx[tid];
    float val = value[tid];
    int src_idx = col * feat_in + threadIdx.x;
    int dst_idx = row * feat_in + threadIdx.x;
    if (feat_in == SMALL_WIDTH) {
        atomicAdd(&vout[dst_idx], vin[src_idx] * val);
    }
    else 
    {
        #pragma unroll 8
        for (int j = 0; j < LARGE_WIDTH; j += BLOCK_LEN)
        {
            atomicAdd(&vout[dst_idx + j], vin[src_idx + j] * val);
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
    num_blocks = (num_e + block_size - 1) / block_size;
    checkCudaErrors(cudaMalloc2((void**)&d_col_idx, num_e * sizeof(int)));
    checkCudaErrors(cudaMalloc2((void**)&d_row_idx, num_e * sizeof(int)));
    checkCudaErrors(cudaMalloc2((void**)&d_value, num_e * sizeof(float)));
    column_major_kernel<<<num_blocks, block_size>>>(d_coo, d_idx, d_val, d_col_idx, d_row_idx, d_value, num_e);

    // Decide grid and block size for spmm_kernel_opt
    block_size = 32;
    grid.x = (num_e + block_size - 1) / block_size;
    block.y = block_size;
    block.x = BLOCK_LEN;
}

void SpMMOpt::run(float *vin, float *vout)
{
    // TODO: your code
    spmm_kernel_opt<<<grid, block>>>(d_col_idx, d_row_idx, d_value, vin, vout, num_v, num_e, feat_in);
}