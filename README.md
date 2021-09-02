# DFToy

3D DFT toy code in C and Fortran.

Performs exact and/or iterative calculation of the Eigenstates of a simple
system, representative of a simple DFT calculation.

# Requirements

## Both versions
 - C/Fortran compiler
 - LAPACK
 - BLAS
 - FFTW

## C version only
 - MPI library/compiler
 - ScaLAPACK (optional)
 - ELPA (optional)

# Building

`cd` into either the C or Fortran folder.

Update the `Makefile` to suit your compiler/MPI/library paths and flags and run
`make`.

# Running

## C

For a basic run with default parameters:

`mpirun -np <num_procs> dftoy`

Run `dftoy --usage` for an overview of the available configuration parameters.

## Fortran

Run `dftoy`.

To change the configuration, change the `num_wavevectors` and `num_states` 
parameters in `Fortran/dftoy.f90` and recompile.

## Running larger simulations

The time (and memory) required to solve the eigenstate problem in 3D by exact
diagonalisation increases _very rapidly_ with the number of wavevectors. It is
recommended to only run the iterative solver (`dftoy --iterative`) for larger
problems.

## Different results between C and Fortran
This is currently being looked into.
