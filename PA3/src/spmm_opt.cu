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

__global__ void spmm_kernel_opt(int *ptr, int *idx, float *val, float *vin, float *vout, int num_v, int INFEATURE, int *coo, int num_e)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_e) return;
    int row = coo[tid];
    int col = idx[tid];
    float value = val[tid];
    for (int j = 0; j < INFEATURE; ++j)
    {
        atomicAdd(&vout[row * INFEATURE + j], vin[col * INFEATURE + j] * value);
    }
}

void SpMMOpt::preprocess(float *vin, float *vout)
{
    // Convert CSR to COO
    checkCudaErrors(cudaMalloc2((void**)&d_coo, num_e * sizeof(int)));
    csr2coo_kernel<<<(num_e + 127) / 128, 128>>>(d_ptr, d_coo, num_v);

    int BLOCK_SIZE = 128;
    grid.x = (num_e + BLOCK_SIZE - 1) / BLOCK_SIZE;
    block.x = BLOCK_SIZE;
}

void SpMMOpt::run(float *vin, float *vout)
{
    // TODO: your code
    spmm_kernel_opt<<<grid, block>>>(d_ptr, d_idx, d_val, vin, vout, num_v, feat_in, d_coo, num_e);
}