#include "gtest/gtest.h"
#include "util.h"
#include "valid.h"
#include "spmm_ref.h"
#include "spmm_opt.h"
#include "spmm_cusparse.h"

class SpMMTest : public testing::Test
{
protected:
    vector<void *> tensor_ptr;
    float *p_in_feat_vec, *p_out_feat_vec, *p_out_feat_vec_ref, *p_value, *p_value2;
    int *gptr2, *gidx2;
    CSR *g, *g2;
    virtual void SetUp()
    {
        p_in_feat_vec = allocate<float>(kNumV * kLen, &tensor_ptr);
        p_out_feat_vec = allocate<float>(kNumV * kLen, &tensor_ptr);
        p_out_feat_vec_ref = allocate<float>(kNumV * kLen, &tensor_ptr);
        p_value = allocate<float>(kNumE, &tensor_ptr);
	
	gptr2 = allocate<int>(kNumV + 1, &tensor_ptr);
	gidx2 = allocate<int>(kNumE, &tensor_ptr);
	p_value2 = allocate<float>(kNumE, &tensor_ptr);
	checkCudaErrors(cudaMemcpy(gptr2, gptr, (kNumV + 1) * sizeof(int), cudaMemcpyDeviceToDevice));
	checkCudaErrors(cudaMemcpy(gidx2, gidx, kNumE * sizeof(int), cudaMemcpyDeviceToDevice));
	checkCudaErrors(cudaMemcpy(p_value2, p_value, kNumE * sizeof(float), cudaMemcpyDeviceToDevice));

        g = new CSR(kNumV, kNumE, gptr, gidx, p_value);
	g2 = new CSR(kNumV, kNumE, gptr2, gidx2, p_value2);
    }
    virtual void TearDown()
    {
        for (auto item : tensor_ptr)
        {
            cudaFree(item);
        }
    }
};

TEST_F(SpMMTest, validation)
{
    SpMMRef *spmmer_ref = new SpMMRef(g, kLen);
    SpMMOpt *spmmer = new SpMMOpt(g2, kLen);
    spmmer_ref->preprocess(p_in_feat_vec, p_out_feat_vec_ref);
    spmmer->preprocess(p_in_feat_vec, p_out_feat_vec);
    checkCudaErrors(cudaMemset(p_out_feat_vec, 0, sizeof(float) * kNumV * kLen));
    checkCudaErrors(cudaMemset(p_out_feat_vec_ref, 0, sizeof(float) * kNumV * kLen));
    spmmer_ref->run(p_in_feat_vec, p_out_feat_vec_ref);
    spmmer->run(p_in_feat_vec, p_out_feat_vec);
    checkCudaErrors(cudaDeviceSynchronize());
    // This ASSERT will fail because your SpMM is not implemented yet
    ASSERT_LT(valid(p_out_feat_vec, p_out_feat_vec_ref, kNumV * kLen), kNumV * kLen / 10000 + 1);
}

TEST_F(SpMMTest, cusparse_performance)
{
    SpMMCuSparse *spmmer = new SpMMCuSparse(g, kLen);
    spmmer->preprocess(p_in_feat_vec, p_out_feat_vec);
    auto time = getAverageTimeWithWarmUp([&]()
                                         { spmmer->run(p_in_feat_vec, p_out_feat_vec); });
    dbg(time);
}

TEST_F(SpMMTest, opt_performance)
{
    SpMMOpt *spmmer = new SpMMOpt(g, kLen);
    spmmer->preprocess(p_in_feat_vec, p_out_feat_vec);
    auto time = getAverageTimeWithWarmUp([&]()
                                         { spmmer->run(p_in_feat_vec, p_out_feat_vec); });
    dbg(time);
}
