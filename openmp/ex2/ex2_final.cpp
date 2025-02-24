#include <cstdio>
#include <omp.h>

int main() {
  const double start_time = omp_get_wtime();
  const int N = 1000000;
  int* array = new int[N];

#pragma omp parallel for default(none) shared(array, N)
  for (int i = 0; i < N; i++) {
    array[i] = i;
  }

  double average = 0.0;

#pragma omp parallel for default(none) reduction(+:average) shared(array, N)
  for (int i = 0; i < N; i++) {
    average += array[i];
  }

  average /= N;

  printf("Average: %f\n", average);
  printf("Total time: %f\n", omp_get_wtime() - start_time);

  delete[] array;
  return 0;
}

