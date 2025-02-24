#include <cstdio>
#include <omp.h>
#include <random>


int main() {
  const double start_time = omp_get_wtime();
  const int N = 1000000;
  int count = 0;

#pragma omp parallel default(none) shared(count, N)
  {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

#pragma omp for reduction(+:count)
    for (int i = 0; i < N; i++) {
      const double x = dis(gen);
      const double y = dis(gen);

      if (x * x + y * y <= 1) {
        count++;
      }
    }
  }

  const double pi = 4.0 * count / N;
  printf("Pi estimate: %f\n", pi);

  printf("Total time: %f seconds\n", omp_get_wtime() - start_time);
  return 0;
}

