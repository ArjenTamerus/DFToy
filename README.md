# DFToy

3D DFT toy code.

Performs exact and/or iterative calculation of the Eigenstates of a simple
system, representative of a simple DFT calculation.

# Requirements
 - C/MPI compiler
 - LAPACK
 - BLAS
 - FFTW
 - ScaLAPACK (optional) - export `WITH_SCALAPACK`
 - ELPA (optional) - export `WITH_ELPA` (implies `WITH_SCALAPACK`)

# Building

Update the `Makefile` to suit your compiler/MPI/library paths and flags and run
`make`.

# Running

For a basic run with default parameters:

`mpirun -np <num_procs> dftoy`

Run `dftoy --usage` for an overview of the available configuration parameters.

## Running larger simulations

The time (and memory) required to solve the eigenstate problem in 3D by exact
diagonalisation increases _very rapidly_ with the number of wavevectors. It is
recommended to only run the iterative solver (`dftoy --iterative`) for larger
problems.

