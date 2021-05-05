#include <stdlib.h>
#include <string.h>
#include <complex.h>
#include <mpi.h>
#include <fftw3.h>
#include <elpa/elpa.h>
#include "interface.h"
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

	return ELPA_2STAGE_COMPLEX_AVX2_BLOCK1;
}

int diag_elpa(int num_pw, double *H_kinetic, double *H_local, double *full_eigenvalue)
{
	fftw_complex *full_H;
	fftw_complex *work;
	double *rwork;
	int *iwork;
	int lwork, lrwork, liwork;
	int i;
	int status;
	char jobz;
	char uplo;

	elpa_t elpa_handle;

	int world_rank;

	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
	mpi_printf(world_rank, "ELPA diagonaliser\n");

	// First we allocate memory for and construct the full Hamiltonian
	full_H = (fftw_complex *)TRACEFFTW_MALLOC(num_pw*num_pw*sizeof(fftw_complex));

	construct_full_H(num_pw,H_kinetic,H_local,full_H);


	int blacs_size, blacs_rank, blacs_ctxt = 0, blacs_ctxt_root = 0, world_size;
	int dims[2] = {0};
	int zero = 0;
	int one = 1;

	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	Cblacs_pinfo(&blacs_rank, &blacs_size);
	MPI_Dims_create(blacs_size, 2, dims);

	int nprow, npcol, myprow, mypcol;
	nprow = dims[0];
	npcol = dims[1];
	Cblacs_get(-1, 0, &blacs_ctxt);
	char cbgi_r = 'R';
	Cblacs_gridinit(&blacs_ctxt, &cbgi_r, nprow, npcol);
	Cblacs_gridinit(&blacs_ctxt_root, &cbgi_r, 1, 1);
	Cblacs_gridinfo(blacs_ctxt, &nprow, &npcol, &myprow, &mypcol);

	//MB = num_pw/nprow;
	int NB = num_pw/nprow;
	int MB = NB;//num_pw/nprow;
	int NLOC_A = numroc_(&num_pw, &NB, &mypcol, &zero, &npcol);
	int MLOC_A = numroc_(&num_pw, &MB, &myprow, &zero, &nprow);

	int LDA = numroc_(&num_pw, &MB, &myprow, &zero, &nprow);
	LDA = LDA < 1 ? 1 : LDA;
	int desc[9]; // apparently DLEN == 9
	int desc_root[9]; // apparently DLEN == 9
	fftw_complex *A = TRACEMALLOC((MLOC_A)*NLOC_A*sizeof(fftw_complex));
	descinit_(desc, &num_pw, &num_pw, &MB, &NB, &zero, &zero, &blacs_ctxt, &LDA, &status);
	if (world_rank == 0) {
		descinit_(desc_root, &num_pw, &num_pw, &num_pw, &num_pw, &zero, &zero, &blacs_ctxt_root, &num_pw, &status);
	} else {
		desc_root[1] = -1;
	}
	pzgemr2d_(&num_pw, &num_pw, full_H, &one, &one, desc_root, A, &one, &one, desc, &blacs_ctxt);

	if(elpa_init(20171201) != ELPA_OK) {
		mpi_printf(world_rank, "UNSUPPORTED ELPA API\n");
		MPI_Abort(MPI_COMM_WORLD, -1);
	}

	elpa_handle = elpa_allocate(&status);
	if (status != ELPA_OK) {
		mpi_printf(world_rank, "ELPA ALLOC FAILED\n");
		MPI_Abort(MPI_COMM_WORLD, -1);
	}

	int elpa_blocksize = 32;

	elpa_set(elpa_handle, "na", num_pw, &status);
	elpa_set(elpa_handle, "nev", num_pw, &status);
	elpa_set(elpa_handle, "local_nrows", MLOC_A, &status);
	elpa_set(elpa_handle, "local_ncols", NLOC_A, &status);
	elpa_set(elpa_handle, "nblk", elpa_blocksize, &status);
	elpa_set(elpa_handle, "mpi_comm_parent", MPI_Comm_c2f(MPI_COMM_WORLD), &status);
	elpa_set(elpa_handle, "process_row", myprow, &status);
	elpa_set(elpa_handle, "process_col", mypcol, &status);
	elpa_set(elpa_handle, "blacs_context", blacs_ctxt, &status);

	status = elpa_setup(elpa_handle);

	int solver = get_elpa_2stage_solver();
	if (solver == ELPA_2STAGE_COMPLEX_GPU) 
		elpa_set(elpa_handle, "gpu", 1, &status);
	elpa_set(elpa_handle, "solver", get_elpa_1_2_stage(), &status);
	elpa_set(elpa_handle, "complex_kernel", solver, &status);


	elpa_eigenvectors(elpa_handle, (double complex*)A, full_eigenvalue, A, &status);

}
