cmake_minimum_required(VERSION 3.12)
project(omp_ex1 CXX)

add_executable(ex1 ex1.cpp)

find_package(OpenMP REQUIRED)
target_link_libraries(ex1 PUBLIC OpenMP::OpenMP_CXX)
