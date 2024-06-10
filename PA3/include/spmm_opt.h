#ifndef SpMM_OPT_H
#define SpMM_OPT_H
#include "spmm_base.h"

class SpMMOpt : public SpMM
{
public:
    SpMMOpt(int *dev_out_ptr, int *dev_out_idx, int out_num_v, int out_num_e, int out_feat_in) : SpMM(dev_out_ptr, dev_out_idx, out_num_v, out_num_e, out_feat_in) {}
    SpMMOpt(CSR *g, int out_feat_in) : SpMM(g, out_feat_in) {}
    ~SpMMOpt() {
        if (d_coo) checkCudaErrors(cudaFree(d_coo));
        if (d_col_idx) checkCudaErrors(cudaFree(d_col_idx));
        if (d_row_idx) checkCudaErrors(cudaFree(d_row_idx));
        if (d_value) checkCudaErrors(cudaFree(d_value));
    }
     
    virtual void preprocess(float *vin, float *vout);

    virtual void run(float *vin, float *vout);

private:
    int *d_coo;
    int *d_col_idx;
    int *d_row_idx;
    float *d_value;
};
#endif