#include <cstdio>
#include <omp.h>

int main()
{
  const double start_time = omp_get_wtime();
  const int N = 1000000;
  int *array = new int[N];
  double average = 0.0;

  printf("Average: %f\n", average);
  printf("Total time: %f\n", omp_get_wtime() - start_time);

  delete[] array;
  return 0;
}
