# 自动向量化与基于intrinsic的手动向量化 实验报告 



管思源 2021012702

## 测试结果

|          | `baseline`<br />(不向量化) | `auto simd`<br />(自动向量化) | `intrinsic`<br />(手动向量化) |
| -------- | -------------------------- | ----------------------------- | ----------------------------- |
| 运行时间 | 4438 us                    | 527 us                        | 526 us                        |
| 加速比   | 1                          | 8.42                          | 8.44                          |

分析：使用`Intel intrinsics`实现向量化的运行时间与编译器自动向量化的相近，说明实现正确。两者加速比均大于预期加速比（8），这可能是利用了`cache`的局部性减少重复读写带来的性能提升。

## 函数代码

```cpp
void a_plus_b_intrinsic(float* a, float* b, float* c, int n) {
    for (int i = 0; i < n; i += 8) {
        __m256 a_vec = _mm256_load_ps(a + i);
        __m256 b_vec = _mm256_load_ps(b + i);
        __m256 c_vec = _mm256_add_ps(a_vec, b_vec);
        _mm256_store_ps(c + i, c_vec);
    }
}
```

