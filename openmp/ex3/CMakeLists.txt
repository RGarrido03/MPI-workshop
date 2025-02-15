cmake_minimum_required(VERSION 3.12)
project(omp_ex3 CXX)

add_executable(ex3 ex3.cpp)

find_package(OpenMP REQUIRED)
target_link_libraries(ex3 PUBLIC OpenMP::OpenMP_CXX)
