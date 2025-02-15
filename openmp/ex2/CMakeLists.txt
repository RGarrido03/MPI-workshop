cmake_minimum_required(VERSION 3.12)
project(omp_ex2 CXX)

add_executable(ex2 ex2.cpp)

find_package(OpenMP REQUIRED)
target_link_libraries(ex2 PUBLIC OpenMP::OpenMP_CXX)
