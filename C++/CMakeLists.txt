cmake_minimum_required(VERSION 3.15)

project(atomistic
  LANGUAGES CXX Fortran)

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

find_package(FFTW REQUIRED)

add_library(hamiltonian
  hamiltonian.f90
  hamiltonian_c.cpp
  hamiltonian_c.hpp)

target_compile_options(hamiltonian PRIVATE
  -Wall -Wextra)

target_link_libraries(hamiltonian PUBLIC FFTW::FFTW)
target_include_directories(hamiltonian PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  )

find_package(LAPACK REQUIRED)

add_executable(eigensolver
  eigensolver.cpp)

target_link_libraries(eigensolver hamiltonian "${LAPACK_LIBRARIES}")
target_compile_features(eigensolver PUBLIC cxx_std_17)
set_target_properties(eigensolver PROPERTIES CXX_EXTENSIONS OFF)
target_compile_options(eigensolver PRIVATE
  -Wall -Wextra)
