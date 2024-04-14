#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <mpi.h>

#include "worker.h"

void radixSort(float* data, int n) {
  float* tmp = new float[n];
  int* cnt = new int[256];
  int* sum = new int[256];
  unsigned char* byte = (unsigned char*)data;
  unsigned char* byte_tmp = (unsigned char*)tmp;
  for (int i = 0; i < 4; i++) {
    memset(cnt, 0, sizeof(int) * 256);
    for (int j = 0; j < n; j++) {
      cnt[byte[j * 4 + i]]++;
    }
    sum[0] = 0;
    for (int j = 1; j < 256; j++) {
      sum[j] = sum[j - 1] + cnt[j - 1];
    }
    for (int j = 0; j < n; j++) {
      tmp[sum[byte[j * 4 + i]]++] = data[j];
    }
    memcpy(data, tmp, sizeof(float) * n);
  }
  delete[] tmp;
  delete[] cnt;
  delete[] sum;
}

void Worker::sort() {
  /** Your code ... */
  // you can use variables in class Worker: n, nprocs, rank, block_len, data
  radixSort(data, block_len);
}