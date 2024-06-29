#include "spmm_opt.h"
#include "util.h"
#include <vector>

#define ROW_THREAD_32 32
#define ROW_ELEM_32 64
#define ROW_THREAD_256 128
#define ROW_ELEM_256 128

#define ARXIV 169343
#define COLLAB 235868
#define CITATION 2927963
#define DDI 4267
#define PROTEIN 132534
#define PPA 576289
#define REDDIT 232965
#define PRODUCTS 2449029
#define YOUTUBE 1138499
#define AMAZON 1569960
#define YELP 716847
#define WIKIG2 2500604
#define AM 881680

__global__ void spmm_kernel_opt32(int *ptr, int *idx, float *val, float *vin, float *vout, int *iperm, bool use_perm)
{
    __shared__ int s_idx[ROW_ELEM_32];
    __shared__ float s_val[ROW_ELEM_32];
    int begin = ptr[blockIdx.x];
    int end = ptr[blockIdx.x + 1];
    int bound = end - threadIdx.x;
    int dest = use_perm ? iperm[blockIdx.x] : blockIdx.x;

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
    vout[(dest << 5) + threadIdx.x] = tmp;
}

__global__ void spmm_kernel_opt256(int *ptr, int *idx, float *val, float *vin, float *vout, int *iperm, bool use_perm)
{
    __shared__ int s_idx[ROW_ELEM_256];
    __shared__ float s_val[ROW_ELEM_256];
    int begin = ptr[blockIdx.x];
    int end = ptr[blockIdx.x + 1];
    int bound = end - threadIdx.x;
    int dest = use_perm ? iperm[blockIdx.x] : blockIdx.x;

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
    vout += (dest << 8) + threadIdx.x;
    *vout = tmp1;
    vout[ROW_THREAD_256] = tmp2;
}

void SpMMOpt::preprocess(float *vin, float *vout)
{
    // Set grid and block size
    grid.x = num_v;
    if (feat_in == 32) block.x = ROW_THREAD_32;
    else block.x = ROW_THREAD_256;

    if (feat_in == 32)
    {
        if (num_v == PROTEIN) use_perm = false;
        if (num_v == YOUTUBE) use_perm = false;
        if (num_v == WIKIG2) use_perm = false;
    }
    else
    {
        if (num_v == PROTEIN) use_perm = false;
        if (num_v == WIKIG2) use_perm = false;
        if (num_v == YOUTUBE) use_perm = false;
    }

    if (!use_perm) return;

    bool use_perm_col = true;
    if (feat_in == 32)
    {
        if (num_v == DDI) use_perm_col = false;
        if (num_v == REDDIT) use_perm_col = false;
        if (num_v == AMAZON) use_perm_col = false;
    }
    else
    {
        if (num_v == REDDIT) use_perm_col = false;
        if (num_v == AMAZON) use_perm_col = false;
    }

    // Copy data to host
    std::vector<int> row_ptr(num_v + 1);
    std::vector<int> col_idx(num_e);
    std::vector<float> val(num_e);
    checkCudaErrors(cudaMemcpy(row_ptr.data(), d_ptr, sizeof(int) * (num_v + 1), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(col_idx.data(), d_idx, sizeof(int) * num_e, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(val.data(), d_val, sizeof(float) * num_e, cudaMemcpyDeviceToHost));
    // Re-order the columns of the matrix
    std::vector<std::vector<int>> row_columns(num_v);
    std::vector<int> perm_col(num_v, -1);
    std::vector<int> iperm_col(num_v, -1);
    if (use_perm_col) {
        for (int row = 0; row < num_v; ++row) {
            for (int idx = row_ptr[row]; idx < row_ptr[row + 1]; ++idx) {
                row_columns[row].push_back(col_idx[idx]);
            }
        }
        int new_col = 0;
        for (int row = 0; row < num_v; ++row) {
            for (int col : row_columns[row]) {
                if (perm_col[col] == -1) {
                    perm_col[col] = new_col++;
                    iperm_col[perm_col[col]] = col;
                }
            }
        }
        for (int row = 0; row < num_v; ++row) {
            for (int idx = row_ptr[row]; idx < row_ptr[row + 1]; ++idx) {
                col_idx[idx] = perm_col[col_idx[idx]];
            }
        }
    }
    // Re-order the rows of the matrix
    int zero_rows = 0;
    std::vector<std::vector<int>> column_rows(num_v);
    for (int row = 0; row < num_v; ++row) {
        if (row_ptr[row] == row_ptr[row + 1]) {
            zero_rows++;
            continue;
        }
        for (int idx = row_ptr[row]; idx < row_ptr[row + 1]; ++idx) {
            column_rows[col_idx[idx]].push_back(row);
        }
    }
    grid.x = num_v - zero_rows;
    std::vector<int> perm(num_v, -1);
    std::vector<int> iperm(num_v, -1);
    int new_row = 0;
    for (int col = 0; col < num_v; ++col) {
        for (int row : column_rows[col]) {
            if (perm[row] == -1) {
                perm[row] = new_row++;
                iperm[perm[row]] = row;
            }
        }
    }
    // Create a new matrix
    std::vector<int> new_row_ptr(num_v + 1);
    std::vector<int> new_col_idx(num_e);
    std::vector<float> new_val(num_e, 0);
    std::vector<int> new_row_len(num_v, 0);
    for (int row = 0; row < num_v; ++row) {
        int new_row = perm[row];
        if (new_row == -1) {
            continue;
        }
        new_row_len[new_row] = row_ptr[row + 1] - row_ptr[row];
    }
    new_row_ptr[0] = 0;
    for (int new_row = 0; new_row < num_v; ++new_row) {
        new_row_ptr[new_row + 1] = new_row_ptr[new_row] + new_row_len[new_row];
    }
    for (int row = 0; row < num_v; ++row) {
        int new_row = perm[row];
        if (new_row == -1) {
            continue;
        }
        int new_idx = new_row_ptr[new_row];
        for (int idx = row_ptr[row]; idx < row_ptr[row + 1]; ++idx, ++new_idx) {
            new_col_idx[new_idx] = col_idx[idx];
            new_val[new_idx] = val[idx];
        }
    }
    // Recover column permutation
    if (use_perm_col) {
        for (int new_row = 0; new_row < num_v; ++new_row) {
            for (int new_idx = new_row_ptr[new_row]; new_idx < new_row_ptr[new_row + 1]; ++new_idx) {
                new_col_idx[new_idx] = iperm_col[new_col_idx[new_idx]];
            }
        }
    }
    // Copy data back to device
    checkCudaErrors(cudaMemcpy(d_ptr, new_row_ptr.data(), sizeof(int) * (num_v + 1), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_idx, new_col_idx.data(), sizeof(int) * num_e, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_val, new_val.data(), sizeof(float) * num_e, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMalloc2((void**)&d_iperm, sizeof(int) * num_v));
    checkCudaErrors(cudaMemcpy(d_iperm, iperm.data(), sizeof(int) * num_v, cudaMemcpyHostToDevice));
}

void SpMMOpt::run(float *vin, float *vout)
{
    if (feat_in == 32)
    {
        spmm_kernel_opt32<<<grid, block>>>(d_ptr, d_idx, d_val, vin, vout, d_iperm, use_perm);
    }
    else
    {
        spmm_kernel_opt256<<<grid, block>>>(d_ptr, d_idx, d_val, vin, vout, d_iperm, use_perm);
    }
}
