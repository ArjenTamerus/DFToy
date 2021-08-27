# DFToy

3D DFT toy code in C and Fortran.

Performs exact and/or iterative calculation of the Eigenstates of a simple
system, representative of a simple DFT calculation.

NOTE: Fortran version is currently 1D (will be updated to support 3D calculation
in the near future)

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

Run `eigensolver`.

To change the configuration, change the `num_wavevectors` and `num_states` in
`Fortran/eigensolver.f90` and recompile.
