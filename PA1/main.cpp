#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <mpi.h>
#include <sys/time.h>

#include "worker.h"

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int nprocs, rank;
  CHKERR(MPI_Comm_size(MPI_COMM_WORLD, &nprocs));
  CHKERR(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
  if (argc != 3) {
    if (!rank)
      printf("Usage: ./odd_even_sort <number_count> <input_file>\n");
    MPI_Finalize();
    return 1;
  }
  const int n = atoi(argv[1]);
  const char *input_name = argv[2];

  if (n < nprocs)
  {
    MPI_Finalize();
    return 0;
  }

  Worker *worker = new Worker(n, nprocs, rank);
  unsigned long *time = new unsigned long[10];

  for (int i = 0; i < 5; i++)
  {
    /** Read input data from the input file */
    worker->input(input_name);
    /** Sort the list (input data) */
    MPI_Barrier(MPI_COMM_WORLD);
    // run your code
    worker->sort();
    MPI_Barrier(MPI_COMM_WORLD);
  }
  for (int i = 0; i < 10; i++)
  {
    /** Read input data from the input file */
    worker->input(input_name);
    /** Sort the list (input data) */
    timeval start, end;
    MPI_Barrier(MPI_COMM_WORLD);
    gettimeofday(&start, NULL);
    // run your code
    worker->sort();
    MPI_Barrier(MPI_COMM_WORLD);
    gettimeofday(&end, NULL);
    time[i] = 1000000.0 * (end.tv_sec - start.tv_sec) + end.tv_usec - start.tv_usec;
  }

  unsigned long avg = 0, min = 0;
  for (int i = 0; i < 10; i++)
  {
    avg += time[i];
    if (time[i] < min || min == 0)
      min = time[i];
  }
  avg /= 10.0;

  /** Check the sorted list */
  int ret = worker->check();
  if (ret > 0) {
#ifndef NDEBUG
    printf("Rank %d: pass\n", rank);
#endif
  } else {
    printf("Rank %d: failed\n", rank);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0) {
    printf("Execution time of function sort is %lf ms.\n", avg / 1000.0);
    printf("Minimum execution time of function sort is %lf ms.\n", min / 1000.0);
  }

#ifndef NDEBUG
  printf("Process %d: finalize\n", rank);
#endif
  MPI_Finalize();
  return 0;
}
