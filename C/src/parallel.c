#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <complex.h>
#include <fftw3.h>
#include "parallel.h"
#include "diag.h"
#include "trace.h"

int world_size, world_rank;
int blacs_ctxt, blacs_ctxt_root;
int nprow, npcol, myprow, mypcol;
bool par_root = false;

void init_parallel(int argc, char **argv)
{
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

	if(!world_rank) {
		par_root = true;
	}

	int blacs_size, blacs_rank;
	int blacs_grid_dims[2] = {0};

	Cblacs_pinfo(&blacs_rank, &blacs_size);
	MPI_Dims_create(blacs_size, 2, blacs_grid_dims);

	nprow = blacs_grid_dims[0];
	npcol = blacs_grid_dims[1];
	Cblacs_get(-1, 0, &blacs_ctxt);
	char cbgi_r = 'R';
	Cblacs_gridinit(&blacs_ctxt, &cbgi_r, nprow, npcol);
	Cblacs_gridinit(&blacs_ctxt_root, &cbgi_r, 1, 1);
	Cblacs_gridinfo(blacs_ctxt, &nprow, &npcol, &myprow, &mypcol);
}

void finalise_parallel()
{
	MPI_Finalize();
}

int mpi_printf(const char *format, ...)
{
	int status = 0;
	va_list myargs;

	if (par_root) {
		va_start(myargs, format);
		status = vprintf(format, myargs);
		va_end(myargs);
	}

	return status;
}

int mpi_error(const char *format, ...)
{
	int status = 0;
	const char template[] = "[%d/%d] (EE) %s";
	char full_format[1024];
	va_list myargs;

	snprintf(full_format, 1024, template, world_rank, world_size, format);

	va_start(myargs, format);
	status = vfprintf(stderr, full_format, myargs);
	va_end(myargs);

	return status;
}

void mpi_fail(const char *format, ...)
{
	va_list myargs;

	if (par_root) {
		va_start(myargs, format);
		mpi_error(format, myargs);
		va_end(myargs);
	}

	MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
}

void distribute_matrix_for_diagonaliser(int num_plane_waves, int desc[9], 
		fftw_complex *matrix, fftw_complex **A, fftw_complex **Z)
{
	int NB;
	int MB;
	int NLOC_A;
	int MLOC_A;

	int LDA;
	int desc_root[9]; // apparently DLEN == 9

	int status;
	int zero = 0;
	int one = 1;

	NB = num_plane_waves/nprow;
	MB = NB;//num_plane_waves/nprow;
	NLOC_A = numroc_(&num_plane_waves, &NB, &mypcol, &zero, &npcol);
	MLOC_A = numroc_(&num_plane_waves, &MB, &myprow, &zero, &nprow);

	LDA = numroc_(&num_plane_waves, &MB, &myprow, &zero, &nprow);
	LDA = LDA < 1 ? 1 : LDA;

	descinit_(desc, &num_plane_waves, &num_plane_waves, &MB, &NB, &zero, &zero, &blacs_ctxt, &LDA, &status);
	if (par_root) {
		descinit_(desc_root, &num_plane_waves, &num_plane_waves, &num_plane_waves, &num_plane_waves, &zero, &zero, &blacs_ctxt_root, &num_plane_waves, &status);
	} else {
		desc_root[1] = -1;
	}

	*A = TRACEMALLOC((MLOC_A)*NLOC_A*sizeof(fftw_complex));
	*Z = TRACEMALLOC((MLOC_A)*NLOC_A*sizeof(fftw_complex));

	// distribute full matrix (from root) to local submatrices
	pzgemr2d_(&num_plane_waves, &num_plane_waves, matrix, &one, &one, desc_root, *A, &one, &one, desc, &blacs_ctxt);

}
