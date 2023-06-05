#ifndef TC_PARALLEL_H
#define TC_PARALLEL_H

#include <stdbool.h>
#include <complex.h>
#include <mpi.h>
#include <omp.h>
#include <fftw3.h>

#define XZ 0
#define XY 1
#define YZ 1

extern int world_size, world_rank;
extern int blacs_ctxt, blacs_ctxt_root;
extern int nprow, npcol, myprow, mypcol;
extern bool par_root;

extern int num_omp_threads;

// Prototypes for netlib routines

int numroc_(int *n, int *nb, int *iproc, int *isrcproc, int *nprocs);
int descinit_(int *desc, int *m, int *n, int *mb, int *nb, int *isrc,
		int *icsrc, int *ictxt, int *lld, int *info);

// BLACS routines
int Cblacs_pinfo(int *mypnum, int *nprocs);
int Cblacs_get(int icontxt, int what, int *val);
int Cblacs_gridinit(int *ictxt, char *order, int nprow, int npcol);
int Cblacs_gridinfo(int ictxt, int *nprow, int *npcol, int *myprow,
		int *mypcol);

void pzgemr2d_(int *m, int *n, fftw_complex *A, int *ia, int *ja, int *desca,
		fftw_complex *B, int *ib, int *jb, int *descb, int *ictxt);

void init_parallel(int argc, char **argv);
void finalise_parallel();

int mpi_printf(const char *format, ...);
int mpi_error(const char *format, ...);
void mpi_fail(const char *format, ...);

void distribute_matrix_for_diagonaliser(int num_plane_waves, int desc[9], 
		fftw_complex *matrix, fftw_complex **A, fftw_complex **Z);

void transpose_for_fftw(fftw_complex *in, fftw_complex *out,
		ptrdiff_t local_start, ptrdiff_t local_count, ptrdiff_t num_plane_waves,
		int direction);

void omp_init();
#endif
