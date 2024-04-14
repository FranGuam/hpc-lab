#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <mpi.h>

#include "worker.h"

void radixSort(float* data, int n) {
  int* count = new int[256];
  int* offset = new int[256];
  int* temp = new int[n];
  int* data_int = (int*)data;
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 256; j++) {
      count[j] = 0;
    }
    for (int j = 0; j < n; j++) {
      count[(data_int[j] >> (i * 8)) & 0xff]++;
    }
    offset[0] = 0;
    for (int j = 1; j < 256; j++) {
      offset[j] = offset[j - 1] + count[j - 1];
    }
    for (int j = 0; j < n; j++) {
      temp[offset[(data_int[j] >> (i * 8)) & 0xff]++] = data_int[j];
    }
    for (int j = 0; j < n; j++) {
      data_int[j] = temp[j];
    }
  }
  delete[] count;
  delete[] offset;
  delete[] temp;
}

void Worker::sort() {
  /** Your code ... */
  // you can use variables in class Worker: n, nprocs, rank, block_len, data
  radixSort(data, block_len);
}