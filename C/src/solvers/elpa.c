#include <stdlib.h>
#include <string.h>
#include <complex.h>
#include <fftw3.h>
#include <elpa/elpa.h>
#include "parallel.h"
#include "interfaces.h"
#include "trace.h"

int get_elpa_1_2_stage()
{
	const char *env = getenv("TOYCODE_ELPA_STAGE");

	if(env) {
		if(!strncmp(env, "ELPA_SOLVER_1STAGE", 18)) {
			return ELPA_SOLVER_1STAGE;
		}
		if(!strncmp(env, "ELPA_SOLVER_2STAGE", 18)) {
			return ELPA_SOLVER_2STAGE;
		}
	}

	return ELPA_SOLVER_1STAGE;
}

int get_elpa_2stage_solver()
{
	const char *env = getenv("TOYCODE_ELPA_2STAGE_KERNEL");

	if(env) {
		if(!strncmp(env, "ELPA_2STAGE_COMPLEX_AVX2_BLOCK1", 31)) {
			return ELPA_2STAGE_COMPLEX_AVX2_BLOCK1;
		}
		if(!strncmp(env, "ELPA_2STAGE_COMPLEX_AVX2_BLOCK2", 31)) {
			return ELPA_2STAGE_COMPLEX_AVX2_BLOCK2;
		}
		if(!strncmp(env, "ELPA_2STAGE_COMPLEX_AVX512_BLOCK1", 33)) {
			return ELPA_2STAGE_COMPLEX_AVX512_BLOCK1;
		}
		if(!strncmp(env, "ELPA_2STAGE_COMPLEX_AVX512_BLOCK2", 33)) {
			return ELPA_2STAGE_COMPLEX_AVX512_BLOCK2;
		}
		if(!strncmp(env, "ELPA_2STAGE_COMPLEX_GPU", 22)) {
			return ELPA_2STAGE_COMPLEX_GPU;
		}
	}

	return ELPA_2STAGE_COMPLEX_AVX2_BLOCK2;
}

int diag_elpa(fftw_complex *full_H, double *eigenvalues, int num_plane_waves)
{
	fftw_complex *work;
	double *rwork;
	int *iwork;
	int lwork, lrwork, liwork;
	int i;
	int status;
	char jobz;
	char uplo;

	elpa_t elpa_handle;

	int zero = 0;
	int one = 1;

	mpi_printf("Performing exact diagonalisation with ELPA...\n");

	//MB = num_plane_waves/nprow;
	int NB = num_plane_waves/nprow;
	int MB = NB;//num_plane_waves/nprow;
	int NLOC_A = numroc_(&num_plane_waves, &NB, &mypcol, &zero, &npcol);
	int MLOC_A = numroc_(&num_plane_waves, &MB, &myprow, &zero, &nprow);

	int desc[9];
	fftw_complex *A, *Z;

	distribute_matrix_for_diagonaliser(num_plane_waves, &desc[0], full_H, &A, &Z);

	if(elpa_init(20171201) != ELPA_OK) {
		mpi_error("UNSUPPORTED ELPA API\n");
		MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
	}

	elpa_handle = elpa_allocate(&status);
	if (status != ELPA_OK) {
		mpi_error("ELPA ALLOC FAILED\n");
		MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
	}

	int elpa_blocksize = 128;

	elpa_set(elpa_handle, "na", num_plane_waves, &status);
	elpa_set(elpa_handle, "nev", num_plane_waves, &status);
	elpa_set(elpa_handle, "local_nrows", MLOC_A, &status);
	elpa_set(elpa_handle, "local_ncols", NLOC_A, &status);
	elpa_set(elpa_handle, "nblk", elpa_blocksize, &status);
	elpa_set(elpa_handle, "mpi_comm_parent", MPI_Comm_c2f(MPI_COMM_WORLD),
			&status);
	elpa_set(elpa_handle, "process_row", myprow, &status);
	elpa_set(elpa_handle, "process_col", mypcol, &status);
	elpa_set(elpa_handle, "blacs_context", blacs_ctxt, &status);
	//elpa_set(elpa_handle, "qr", 1, &status);

	status = elpa_setup(elpa_handle);

	int solver = get_elpa_2stage_solver();
	if (solver == ELPA_2STAGE_COMPLEX_GPU) {
		elpa_set(elpa_handle, "gpu", 1, &status);
	}

	elpa_set(elpa_handle, "solver", get_elpa_1_2_stage(), &status);
	elpa_set(elpa_handle, "complex_kernel", solver, &status);

	elpa_eigenvectors(elpa_handle, A, eigenvalues, A,
			&status);

	TRACEFREE(A);
	TRACEFREE(Z);

}
