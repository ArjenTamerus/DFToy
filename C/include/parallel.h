#ifndef TOYCODE_PARALLEL_H
#define TOYCODE_PARALLEL_H

#include <stdbool.h>
#include <complex.h>
#include <mpi.h>
#include <fftw3.h>

extern int world_size, world_rank;
extern int blacs_ctxt, blacs_ctxt_root;
extern int nprow, npcol, myprow, mypcol;
extern bool par_root;

void init_parallel(int argc, char **argv);
void finalise_parallel();

int mpi_printf(const char *format, ...);
int mpi_error(const char *format, ...);

void distribute_matrix_for_diagonaliser(int num_plane_waves, int desc[9], 
		fftw_complex *matrix, fftw_complex **A, fftw_complex **Z);
#endif
