#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <mpi.h>
#include <omp.h>

#include "worker.h"

#define MIN_SORT_SIZE 4
#define SCALAR_TAG 99
#define DATA_TAG 100

int merge(float* recv_buf, int recv_len, float* comp_buf, int slice_num, float* result) {
  int count = 0;
  float* recv_end = recv_buf + recv_len;
  float* comp_end = comp_buf + slice_num;
  // TODO: 并行化
  for (int i = 0; i < recv_len; i++) {
    if (comp_buf == comp_end) {
      *result++ = *recv_buf++;
    }
    else if (*recv_buf <= *comp_buf) {
      *result++ = *recv_buf++;
    } else {
      *result++ = *comp_buf++;
      count++;
    }
  }
  while (true) {
    if (recv_buf == recv_end) {
      std::copy(comp_buf, comp_end, result);
      return count;
    }
    if (comp_buf == comp_end) {
      std::copy(recv_buf, recv_end, result);
      return count;
    }
    *result++ = (*comp_buf < *recv_buf)? *comp_buf++ : *recv_buf++;
  }
}

void Worker::sort() {
  /** Your code ... */
  // you can use variables in class Worker: n, nprocs, rank, block_len, data
  bool finished = out_of_range;
  bool all_finished = false;
  int current_sort_size = MIN_SORT_SIZE;
  int upward_count = 0;
  int downward_count = 0;
  int slice_num = block_len / current_sort_size;
  int send_len = 0;
  int recv_len = 0;
  float* send_buf = nullptr;
  float* comp_buf = nullptr;
  float* data_buf = nullptr;
  float* recv_buf = nullptr;
  MPI_Request req;

  #pragma omp parallel for schedule(guided)
  for (int i = 0; i < slice_num; i++) {
    // TODO: 可以先冒泡最大值、再冒泡最小值，剩余再排序
    if (i == slice_num - 1) {
      std::sort(data + i * current_sort_size, data + block_len);
      for (int j = i * current_sort_size; j < block_len; j++) {
        std::cout << data[j] << " ";
      }
      std::cout << std::endl;
    }
    else {
      std::sort(data + i * current_sort_size, data + (i + 1) * current_sort_size);
    }
  }

  while (!all_finished) {
    if (!last_rank) {
      MPI_Isend(&slice_num, 1, MPI_INT, rank + 1, SCALAR_TAG, MPI_COMM_WORLD, &req);
      send_len = slice_num;
      send_buf = new float[send_len];
      #pragma omp parallel for schedule(guided)
      for (int i = 0; i < slice_num; i++) {
        if (i == slice_num - 1)
          send_buf[i] = data[block_len - 1];
        else
          send_buf[i] = data[(i + 1) * current_sort_size - 1];
      }
      MPI_Wait(&req, nullptr);
      MPI_Isend(send_buf, send_len, MPI_FLOAT, rank + 1, DATA_TAG, MPI_COMM_WORLD, &req);
      delete[] send_buf;
    }
    if (rank) {
      if (out_of_range) {
        MPI_Recv(&recv_len, 1, MPI_INT, rank - 1, SCALAR_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        recv_buf = new float[recv_len];
        MPI_Recv(recv_buf, recv_len, MPI_FLOAT, rank - 1, DATA_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        send_buf = new float[recv_len + 1];
        send_buf[0] = 0.0;
        std::copy(recv_buf, recv_buf + recv_len, send_buf + 1);
        delete[] recv_buf;
      }
      else {
        comp_buf = new float[slice_num];
        #pragma omp parallel for schedule(guided)
        for (int i = 0; i < slice_num; i++) {
          comp_buf[i] = data[i * current_sort_size];
        }
        // TODO: 只需分堆
        std::sort(comp_buf, comp_buf + slice_num);
        MPI_Recv(&recv_len, 1, MPI_INT, rank - 1, SCALAR_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        recv_buf = new float[recv_len];
        MPI_Recv(recv_buf, recv_len, MPI_INT, rank - 1, DATA_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        // TODO: 只需分堆
        std::sort(recv_buf, recv_buf + recv_len);
        send_buf = new float[recv_len + slice_num + 1];
        downward_count = merge(recv_buf, recv_len, comp_buf, slice_num, send_buf + 1);
        send_buf[0] = (float)downward_count;
        delete[] comp_buf;
        delete[] recv_buf;
      }
    }
    if (!last_rank) {
      MPI_Wait(&req, nullptr);
    }
    if (rank) {
      send_len = recv_len + 1;
      MPI_Isend(send_buf, send_len, MPI_FLOAT, rank - 1, DATA_TAG, MPI_COMM_WORLD, &req);
      if (!out_of_range) {
        #pragma omp parallel for schedule(guided)
        for (int i = 0; i < slice_num; i++) {
          data[i * current_sort_size] = send_buf[i + send_len];
          // 冒泡
          if (i == slice_num - 1) {
            for (int j = i * current_sort_size; j < block_len; j++) {
              if (data[j] < data[j + 1]) break;
              std::swap(data[j], data[j + 1]);
            }
          }
          else {
            for (int j = i * current_sort_size; j < (i + 1) * current_sort_size - 1; j++) {
              if (data[j] < data[j + 1]) break;
              std::swap(data[j], data[j + 1]);
            }
          }
        }
      }
      delete[] send_buf;
    }
    if (!last_rank) {
      recv_len = slice_num + 1;
      recv_buf = new float[recv_len];
      MPI_Recv(recv_buf, recv_len, MPI_FLOAT, rank + 1, DATA_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      upward_count = (int)recv_buf[0];
      #pragma omp parallel for schedule(guided)
      for (int i = 0; i < slice_num; i++) {
        if (i == slice_num - 1) {
          data[block_len - 1] = recv_buf[i + 1];
          // 冒泡
          for (int j = block_len - 1; j > (slice_num - 1) * current_sort_size; j--) {
            if (data[j] > data[j - 1]) break;
            std::swap(data[j], data[j - 1]);
          }
        }
        else {
          data[(i + 1) * current_sort_size - 1] = recv_buf[i + 1];
          // 冒泡
          for (int j = (i + 1) * current_sort_size - 1; j > i * current_sort_size; j--) {
            if (data[j] > data[j - 1]) break;
            std::swap(data[j], data[j - 1]);
          }
        }
      }
      delete[] recv_buf;
    }
    finished = upward_count == 0 && downward_count == 0 && slice_num == 1;
    if (!finished && downward_count < slice_num / 2 && upward_count < slice_num / 2) {
      if (rank == 2) std::cout << "Entered" << std::endl;
      data_buf = new float[block_len];
      #pragma omp parallel for schedule(guided)
      for (int i = 0; i < slice_num - 1; i += 2) {
        if (i == slice_num - 2) {
          std::merge(
            data + i * current_sort_size, data + (i + 1) * current_sort_size,
            data + (i + 1) * current_sort_size, data + block_len,
            data_buf + i * current_sort_size
          );
          std::copy(data_buf + i * current_sort_size, data_buf + block_len, data + i * current_sort_size);
        }
        else {
          std::merge(
            data + i * current_sort_size, data + (i + 1) * current_sort_size,
            data + (i + 1) * current_sort_size, data + (i + 2) * current_sort_size,
            data_buf + i * current_sort_size
          );
          std::copy(data_buf + i * current_sort_size, data_buf + (i + 2) * current_sort_size, data + i * current_sort_size);
        }
      }
      if (slice_num % 2 == 1) {
        std::merge(
          data + (slice_num - 3) * current_sort_size, data + (slice_num - 1) * current_sort_size,
          data + (slice_num - 1) * current_sort_size, data + block_len,
          data_buf + (slice_num - 3) * current_sort_size
        );
        std::copy(data_buf + (slice_num - 3) * current_sort_size, data_buf + block_len, data + (slice_num - 3) * current_sort_size);
      }
      current_sort_size *= 2;
      slice_num = block_len / current_sort_size;
      delete[] data_buf;
    }
    // TODO: Ring Reduce
    MPI_Allreduce(&finished, &all_finished, 1, MPI_C_BOOL, MPI_LAND, MPI_COMM_WORLD);
    if (rank) {
      MPI_Wait(&req, nullptr);
    }
    std::cout << "Rank " << rank << ": Size " << current_sort_size << " , Slice " << slice_num << " , Upward " << upward_count << " , Downward " << downward_count << std::endl;
    if (rank == 3) {
      for (int i = 0; i < (int)block_len; i++) {
        std::cout << data[i] << " ";
      }
      std::cout << std::endl;
    }
  }
}
