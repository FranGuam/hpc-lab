#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <mpi.h>
#include <string.h>

#include "worker.h"

void radixSort(float* data, int n) {
  int* count = new int[256];
  int* offset = new int[256];
  int* temp = new int[n];
  int* data_int = (int*)data;

  // 0-7
  memset(count, 0, sizeof(int) * 256);
  for (int j = 0; j < n; j++) {
    count[data_int[j] & 0xff]++;
  }
  offset[0] = 0;
  for (int j = 1; j < 256; j++) {
    offset[j] = offset[j - 1] + count[j - 1];
  }
  for (int j = 0; j < n; j++) {
    temp[offset[data_int[j] & 0xff]++] = data_int[j];
  }

  // 8-15
  memset(count, 0, sizeof(int) * 256);
  for (int j = 0; j < n; j++) {
    count[(temp[j] >> 8) & 0xff]++;
  }
  offset[0] = 0;
  for (int j = 1; j < 256; j++) {
    offset[j] = offset[j - 1] + count[j - 1];
  }
  for (int j = 0; j < n; j++) {
    data_int[offset[(temp[j] >> 8) & 0xff]++] = temp[j];
  }

  // 16-23
  memset(count, 0, sizeof(int) * 256);
  for (int j = 0; j < n; j++) {
    count[(data_int[j] >> 16) & 0xff]++;
  }
  offset[0] = 0;
  for (int j = 1; j < 256; j++) {
    offset[j] = offset[j - 1] + count[j - 1];
  }
  for (int j = 0; j < n; j++) {
    temp[offset[(data_int[j] >> 16) & 0xff]++] = data_int[j];
  }

  // 24-31
  memset(count, 0, sizeof(int) * 256);
  for (int j = 0; j < n; j++) {
    count[(temp[j] >> 24) & 0xff]++;
  }
  offset[0] = 0;
  for (int j = 1; j < 256; j++) {
    offset[j] = offset[j - 1] + count[j - 1];
  }
  for (int j = 0; j < n; j++) {
    data_int[offset[(temp[j] >> 24) & 0xff]++] = temp[j];
  }

  delete[] count;
  delete[] offset;
  delete[] temp;
}

int merge(float* first1, float* last1, float* first2, float* last2, float* result) {
  int count = 0;
  while (true) {
    if (first1 == last1) {
      memcpy(result, first2, sizeof(float) * (last2 - first2));
      break;
    }
    if (first2 == last2) {
      memcpy(result, first1, sizeof(float) * (last1 - first1));
      break;
    }
    if (*first2 < *first1) {
      *result++ = *first2++;
      count++;
    } else {
      *result++ = *first1++;
    }
  }
  return count;
}

void Worker::sort() {
  /** Your code ... */
  // you can use variables in class Worker: n, nprocs, rank, block_len, data
  radixSort(data, block_len);
  if (nprocs == 1) return;

  int block_size = ceiling(n, nprocs);
  int first_half = (block_len + 1) / 2;
  int second_half = block_size / 2;
  float* recv_buf = new float[first_half];
  float* send_buf = new float[first_half + second_half];
  MPI_Request request;

  std::cout << "Rank: " << rank << ", First: " << first_half << ", Second: " << second_half << std::endl;

  for (int i = 0; i < nprocs * 2; i++) {
    if (!last_rank) {
      if (i) MPI_Wait(&request, nullptr);
      MPI_Isend(data, second_half, MPI_FLOAT, rank + 1, rank, MPI_COMM_WORLD, &request);
    }
    if (rank) {
      memset(send_buf, 0, sizeof(float) * (first_half + second_half));
      MPI_Recv(recv_buf, second_half, MPI_FLOAT, rank - 1, rank - 1, MPI_COMM_WORLD, nullptr);
      if (last_rank) {
        for (int j = 0; j < (int)block_len; j++) {
          std::cout << data[j] << " ";
        }
        std::cout << std::endl;
        for (int j = 0; j < second_half; j++) {
          std::cout << recv_buf[j] << " ";
        }
        std::cout << std::endl;
      }
      int count = merge(recv_buf, recv_buf + second_half, data, data + first_half, send_buf);
      memcpy(data, send_buf + second_half, sizeof(float) * first_half);
      if (!last_rank) MPI_Wait(&request, nullptr);
      MPI_Isend(send_buf, second_half, MPI_FLOAT, rank - 1, rank, MPI_COMM_WORLD, &request);
      std::cout << "Iter: " << i << ", Rank: " << rank << ", Count: " << count << std::endl;
    }
    if (!last_rank) {
      MPI_Recv(data + (block_len + 1) / 2, block_len / 2, MPI_FLOAT, rank + 1, rank + 1, MPI_COMM_WORLD, nullptr);
    }
    std::inplace_merge(data, data + (block_len + 1) / 2, data + block_len);
  }

  delete[] recv_buf;
  delete[] send_buf;
  MPI_Wait(&request, nullptr);
}