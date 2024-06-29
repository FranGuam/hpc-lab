#include "spmm_opt.h"
#include "util.h"
#include <vector>

#define ROW_NUM_32 1
#define ROW_THREAD_32 32
#define ROW_ELEM_32 64
#define ROW_NUM_256 1
#define ROW_THREAD_256 128
#define ROW_ELEM_256 128

__global__ void spmm_kernel_opt32(int *ptr, int *idx, float *val, float *vin, float *vout, int *iperm)
{
    __shared__ int s_idx[ROW_NUM_32 * ROW_ELEM_32];
    __shared__ float s_val[ROW_NUM_32 * ROW_ELEM_32];
    // int tid = blockIdx.x * blockDim.y + threadIdx.y;
    // if (tid >= num_v) return;
    int begin = ptr[blockIdx.x];
    int end = ptr[blockIdx.x + 1];
    int bound = end - threadIdx.x;
    // int offset = threadIdx.y * ROW_ELEM_32;
    // int *s_idx_base = s_idx + offset;
    // float *s_val_base = s_val + offset;
    int *s_idx_base = s_idx + threadIdx.x;
    float *s_val_base = s_val + threadIdx.x;
    float *vin_base = vin + threadIdx.x;
    int *idx_base = idx + threadIdx.x;
    float *val_base = val + threadIdx.x;

    float tmp = 0;
    for (int i = begin; i < end; i += ROW_ELEM_32)
    {
        // Load data into shared memory
        if (i < bound)
        {
            *s_idx_base = idx_base[i];
            *s_val_base = val_base[i];
        }
        int ii = i + ROW_THREAD_32;
        if (ii < bound)
        {
            s_idx_base[ROW_THREAD_32] = idx_base[ii];
            s_val_base[ROW_THREAD_32] = val_base[ii];
        }
        __syncwarp();

        // Compute
        int max = min(ROW_ELEM_32, end - i);
        for (int j = 0; j < max; ++j)
        {
            tmp += vin_base[(s_idx[j] << 5)] * s_val[j];
        }
        __syncwarp();
    }
    // vout[(tid << 5) + threadIdx.x] = tmp;
    vout[(iperm[blockIdx.x] << 5) + threadIdx.x] = tmp;
}

__global__ void spmm_kernel_opt256(int *ptr, int *idx, float *val, float *vin, float *vout, int *iperm)
{
    __shared__ int s_idx[ROW_NUM_256 * ROW_ELEM_256];
    __shared__ float s_val[ROW_NUM_256 * ROW_ELEM_256];
    // int tid = blockIdx.x * blockDim.y + threadIdx.y;
    // if (tid >= num_v) return;
    // int begin = ptr[tid], end = ptr[tid + 1];
    int begin = ptr[blockIdx.x];
    int end = ptr[blockIdx.x + 1];
    int bound = end - threadIdx.x;
    // int offset = threadIdx.y * ROW_ELEM_256;
    // int *s_idx_base = s_idx + offset;
    // float *s_val_base = s_val + offset;
    int *s_idx_base = s_idx + threadIdx.x;
    float *s_val_base = s_val + threadIdx.x;
    float *vin_base1 = vin + threadIdx.x;
    float *vin_base2 = vin_base1 + ROW_THREAD_256;
    int *idx_base = idx + threadIdx.x;
    float *val_base = val + threadIdx.x;

    float tmp1 = 0, tmp2 = 0;
    for (int i = begin; i < end; i += ROW_ELEM_256)
    {
        // Load data into shared memory
        if (i < bound)
        {
            *s_idx_base = idx_base[i];
            *s_val_base = val_base[i];
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
    vout += (iperm[blockIdx.x] << 8) + threadIdx.x;
    *vout = tmp1;
    vout[ROW_THREAD_256] = tmp2;
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

    // Copy data to host
    std::vector<int> row_ptr(num_v + 1);
    std::vector<int> col_idx(num_e);
    std::vector<float> val(num_e);
    checkCudaErrors(cudaMemcpy(row_ptr.data(), d_ptr, sizeof(int) * (num_v + 1), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(col_idx.data(), d_idx, sizeof(int) * num_e, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(val.data(), d_val, sizeof(float) * num_e, cudaMemcpyDeviceToHost));
    // Re-order the rows of the matrix
    std::vector<std::vector<int>> columns(num_v);
    for (int row = 0; row < num_v; ++row) {
        for (int idx = row_ptr[row]; idx < row_ptr[row + 1]; ++idx) {
            columns[col_idx[idx]].push_back(row);
        }
    }
    std::vector<int> perm(num_v, -1);
    std::vector<int> iperm(num_v, -1);
    int new_row = 0;
    for (int col = 0; col < num_v; ++col) {
        for (int row : columns[col]) {
            if (perm[row] == -1) {
                perm[row] = new_row++;
                iperm[perm[row]] = row;
            }
        }
    }
    // Create a new matrix
    std::vector<int> new_row_ptr(num_v + 1);
    std::vector<int> new_col_idx(num_e);
    std::vector<float> new_val(num_e);
    std::vector<int> new_row_len(num_v);
    for (int row = 0; row < num_v; ++row) {
        int new_row = perm[row];
        new_row_len[new_row] = row_ptr[row + 1] - row_ptr[row];
    }
    new_row_ptr[0] = 0;
    for (int new_row = 0; new_row < num_v; ++new_row) {
        new_row_ptr[new_row + 1] = new_row_ptr[new_row] + new_row_len[new_row];
    }
    for (int row = 0; row < matrix.num_rows; ++row) {
        int new_row = perm[row];
        int new_idx = new_row_ptr[new_row];
        for (int idx = row_ptr[row]; idx < row_ptr[row + 1]; ++idx, ++new_idx) {
            new_col_idx[new_idx] = col_idx[idx];
            new_val[new_idx] = val[idx];
        }
    }
    // Copy data back to device
    checkCudaErrors(cudaMemcpy(d_ptr, new_row_ptr.data(), sizeof(int) * (num_v + 1), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_idx, new_col_idx.data(), sizeof(int) * num_e, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_val, new_val.data(), sizeof(float) * num_e, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMalloc2(&d_iperm, sizeof(int) * num_v));
    checkCudaErrors(cudaMemcpy(d_iperm, iperm.data(), sizeof(int) * num_v, cudaMemcpyHostToDevice));
}

void SpMMOpt::run(float *vin, float *vout)
{
    if (feat_in == 32)
    {
        spmm_kernel_opt32<<<grid, block>>>(d_ptr, d_idx, d_val, vin, vout, d_iperm);
    }
    else
    {
        spmm_kernel_opt256<<<grid, block>>>(d_ptr, d_idx, d_val, vin, vout, d_iperm);
    }
}
