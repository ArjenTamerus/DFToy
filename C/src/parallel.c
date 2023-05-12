/*
 * Parallel.c
 *
 * Support routines facilitating parallel compute/communication.
 *
 */

#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <complex.h>
#include <fftw3-mpi.h>
#include "parallel.h"
#include "trace.h"

int world_size, world_rank;
int blacs_ctxt, blacs_ctxt_root;
int nprow, npcol, myprow, mypcol;
bool par_root = false;

int num_omp_threads = 1;

// Initialise MPI and BLACS
void init_parallel(int argc, char **argv)
{
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

	fftw_mpi_init();

	if(!world_rank) {
		par_root = true;
	}

#ifdef DFTOY_USE_SCALAPACK 
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
#endif

	omp_init();
}

void finalise_parallel()
{
	fftw_mpi_cleanup();
	MPI_Finalize();
}

// Root prints message to stdout
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

// Root prints message to stderr
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


// Root prints to stderr, everyone aborts - fatal error.
void mpi_fail(const char *format, ...)
{
	va_list myargs;

	if (par_root) {
		va_start(myargs, format);
		mpi_error(format, myargs);
		va_end(myargs);
	}

	finalise_parallel();
	exit(EXIT_FAILURE);
}

void distribute_matrix_for_diagonaliser(int num_plane_waves, int desc[9], 
		fftw_complex *matrix, fftw_complex **A, fftw_complex **Z)
{
#ifdef DFTOY_USE_SCALAPACK
	int NB;
	int MB;
	int NLOC_A;
	int MLOC_A;

	int LDA;
	int desc_root[9]; // apparently DLEN == 9

	int status;
	int zero = 0;
	int one = 1;

	// BLACS set-up
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
#endif
}

void transpose_for_fftw(fftw_complex *in, fftw_complex *out,
		ptrdiff_t local_start, ptrdiff_t local_z, ptrdiff_t num_plane_waves,
		int direction)
{
	size_t x, y, z;
	size_t nb, count;
	int rank;
	int xdim_nblocks = world_size;
	ptrdiff_t xdim_block = local_z;
	ptrdiff_t ydim = num_plane_waves;

	static MPI_Request transpose_req = MPI_REQUEST_NULL;


	// NOTE: These are not _necessary_ - can reuse in and out if they're allocated
	// to the appropriate size. To keep the code more readable though, this should
	// only be done if profiling demonstrates an *actual* speed-up or memory
	// savings.
	//
	// Very rough estimate from watching _htop_ says - maybe 5% memory savings at
	// -s 24 -w 24? No noticable speed diff.
	// Would have to test for larger inputs, and run some profiles.
	static fftw_complex *send = NULL, *recv = NULL;

	if (direction == XZ) {

		if (transpose_req == MPI_REQUEST_NULL) {
			mpi_printf("Initialising AlltoAll\n");
			send = calloc(local_z*ydim*(xdim_nblocks*xdim_block), sizeof(fftw_complex));
			recv = calloc(local_z*ydim*(xdim_nblocks*xdim_block), sizeof(fftw_complex));
			MPI_Alltoall_init(send, local_z*ydim*xdim_block*2, MPI_DOUBLE,
				recv, local_z*ydim*xdim_block*2, MPI_DOUBLE, MPI_COMM_WORLD,
				MPI_INFO_NULL, &transpose_req);
		}

		count = 0;
		for(nb = 0; nb < xdim_nblocks; nb++) {
			for(z = 0; z < local_z; z++) {
				for(y = 0; y < ydim; y++) {
					for(x = 0; x < xdim_block; x++) {
						if((nb*xdim_block+x >= num_plane_waves)) {
							send[count] = 0;
							count++;
						}
						else {
							send[count] = in[nb*xdim_block+z*ydim*ydim
								+ y*ydim + x];
							count++;
						}
					}
				}
			}
		}

		MPI_Start(&transpose_req);
		MPI_Wait(&transpose_req, MPI_STATUS_IGNORE);

		count = 0;
		for(nb = 0; nb < xdim_nblocks; nb++) {
			for(z = 0; z < local_z; z++) {
				for(y = 0; y < ydim; y++) {
					for(x = 0; x < xdim_block; x++) {
						if((nb*xdim_block+z >= num_plane_waves)) {
							count++;
						}
						else {
							 out[nb*xdim_block+x*ydim*ydim+y*ydim+z] = recv[count];
							 count++;
						}
					}
				}
			}
		}

		//free(send);
		//free(recv);
	}
	else {
		mpi_error("Invalid transpose direction.\n!");
	}
}

void omp_init()
{
	num_omp_threads = omp_get_num_threads();
}
