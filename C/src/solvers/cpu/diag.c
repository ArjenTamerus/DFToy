#include <stdlib.h>
#include <string.h>
#include <complex.h>
#include <math.h>
#include <time.h>
#include <mpi.h>
#include <fftw3.h>
#include <lapacke.h>
#include "interfaces.h"
#include "trace.h"

int get_diag_mode()
{
	int mode = 0;
	const char *zheev_env = getenv("TOYCODE_DIAG");

	if(zheev_env) {
		if(!strncmp(zheev_env, "ZHEEV", 5)) {
			mode = 0;
		}
		if(!strncmp(zheev_env, "ZHEEVD", 6)) {
			mode = 1;
		}
		if(!strncmp(zheev_env, "ZHEEVR", 6)) {
			if(!strncmp(zheev_env, "ZHEEVR_I", 8)) {
				mode = 6;
			}
			mode = 2;
		}
		if(!strncmp(zheev_env, "PZHEEV", 6)) {
			mode = 3;
		}
		if(!strncmp(zheev_env, "PZHEEVD", 7)) {
			mode = 4;
		}
		if(!strncmp(zheev_env, "PZHEEVR", 7)) {
			if(!strncmp(zheev_env, "PZHEEVR_I", 9)) {
				mode = 7;
			}
			else {
				mode = 5;
			}
		}
		if(!strncmp(zheev_env, "ELPA", 4)) {
			mode = 8;
		}
	}

	return mode;
}

void diag_abort_on_error(int errcode, const char *diag_type)
{
	int rank;
	if(errcode) {
		MPI_Comm_rank(MPI_COMM_WORLD, &rank);
		mpi_printf(rank, "(EE) Error in %p diagonalisation: %d\n", errcode, diag_type);
		MPI_Abort(MPI_COMM_WORLD, errcode);
	}
}

void diag_zheev(fftw_complex *full_H, double *eigenvalues, int num_plane_waves)
{
	int err;

	printf("Performing exact diagonalisation with ZHEEV...\n");

	err = LAPACKE_zheev(LAPACK_COL_MAJOR, 'V', 'U', num_plane_waves, full_H,
			num_plane_waves, eigenvalues);

	diag_abort_on_error(err, "ZHEEV");
}

void diag_zheevd(fftw_complex *full_H, double *eigenvalues, int num_plane_waves)
{
	int err;

	printf("Performing exact diagonalisation with ZHEEVD...\n");

	err = LAPACKE_zheevd(LAPACK_COL_MAJOR, 'V', 'U', num_plane_waves, full_H,
			num_plane_waves, eigenvalues);

	diag_abort_on_error(err, "ZHEEVD");
}


void diag_zheevr(fftw_complex *full_H, double *eigenvalues, int num_plane_waves,
		int num_states)
{
	char jobz = 'V';
	char uplo = 'U';

	char range = 'I'; // Only find IL-th through IU-th eigenvalues
	int IL = 1;
	int IU = num_states;

	// range == I -> VL and BU are not referenced, so set to 0
	double VL = 0.0;
	double VU = 0.0;

	// ABSTOL - need to figure out if we want to play with this, set to 0.0 (==
	// default tolerance) for now
	double abstol = 0.0;

	// zheevr-specific out-params
	// M-param
	int eigenvals_found;

	int *isuppz = TRACECALLOC(2*(IU-IL+1), sizeof(int));

	// zheevr-specific arrays and descriptors
	int ldZ = num_plane_waves;
	fftw_complex *z_work = TRACEFFTW_MALLOC(ldZ*num_states*sizeof(fftw_complex));

	int err = LAPACKE_zheevr(LAPACK_COL_MAJOR, jobz, range, uplo, num_plane_waves,
			full_H, num_plane_waves, VL, VU, IL, IU, abstol, &eigenvals_found,
			eigenvalues, z_work, ldZ, isuppz);

	printf("Performing exact diagonalisation with ZHEEVR...\n");

	diag_abort_on_error(err, "ZHEEVR");

}

void diag_pzheev(fftw_complex *full_H, double *eigenvalues, int num_plane_waves)
{
	mpi_printf(0, "LAPACK diagonaliser:  PZHEEV\n");
	fftw_complex *work;
	double *rwork;
	int lwork, lrwork;
	int i,j;
	int status;
	char jobz;
	char uplo;

	int blacs_size, blacs_rank, blacs_ctxt = 0, blacs_ctxt_root = 0, world_rank, world_size;
	int dims[2] = {0};
	int zero = 0;
	int one = 1;

	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
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

	int NB = num_plane_waves/nprow;
	int MB = NB;
	int NLOC_A = numroc_(&num_plane_waves, &NB, &mypcol, &zero, &npcol);
	int MLOC_A = numroc_(&num_plane_waves, &MB, &myprow, &zero, &nprow);

	int LDA = numroc_(&num_plane_waves, &MB, &myprow, &zero, &nprow);
	LDA = LDA < 1 ? 1 : LDA;

	int desc[9]; // apparently DLEN == 9
	int desc_root[9]; // apparently DLEN == 9
	descinit_(desc, &num_plane_waves, &num_plane_waves, &MB, &NB, &zero, &zero, &blacs_ctxt, &LDA, &status);

	if (world_rank == 0) {
		descinit_(desc_root, &num_plane_waves, &num_plane_waves, &num_plane_waves, &num_plane_waves, &zero, &zero, &blacs_ctxt_root, &num_plane_waves, &status);
	} else {
		desc_root[1] = -1;
	}

	fftw_complex *A = TRACEMALLOC(MLOC_A*NLOC_A*sizeof(fftw_complex));
	fftw_complex *Z = TRACEMALLOC(MLOC_A*NLOC_A*sizeof(fftw_complex));

	// distribute full matrix (from root) to local submatrices
	pzgemr2d_(&num_plane_waves, &num_plane_waves, full_H, &one, &one, desc_root, A, &one, &one, desc, &blacs_ctxt);

	//// LWORK
	int NP0 = numroc_(&num_plane_waves, &MB, &zero, &zero, &nprow);
	int NQ0 = numroc_(&num_plane_waves, &MB, &zero, &zero, &npcol); // num_plane_waves always >= 2 and >= NB

	// eigenvalues + eigenvectors
	lwork = (NP0+NQ0+NB)*NB + 3*num_plane_waves + num_plane_waves*num_plane_waves;
	work = TRACEFFTW_MALLOC(lwork*sizeof(fftw_complex));

	// values + vectors
	lrwork = 2*(num_plane_waves+num_plane_waves)-2;
	rwork = TRACECALLOC(lrwork,sizeof(double));

	//lwork = -1; lrwork = -1;
	//fftw_complex wsize;
	//double rwsize;
	// Use LAPACK to get eigenvalues and eigenvectors, e.g. the zheev routine
	// NB H is Hermitian (but not packed)
	jobz = 'V';
	uplo = 'U';

	// NOTE: PZHEEV workspace query is broken in Netlib Scalapack <= 2.1

	//// Workspace query
	//pzheev_(&jobz, &uplo, &num_plane_waves, A, &one, &one, desc, eigenvalues, Z, &one, &one, desc, &wsize, &lwork, &rwsize, &lrwork, &status);
	//mpi_printf(0, "Work Query results A: %f+%fi\t%f\n", creal(wsize), cimag(wsize), rwsize);

	//// Allocate work arrays to recommended size
	//lwork = (int)creal(wsize);
	//lrwork = (int)rwsize;

	//work = TRACEFFTW_MALLOC(lwork*sizeof(fftw_complex));
	//rwork = TRACECALLOC(lrwork,sizeof(double));
	

	// Actual work
	pzheev_(&jobz, &uplo, &num_plane_waves, A, &one, &one, desc, eigenvalues, Z, &one, &one, desc, work, &lwork, rwork, &lrwork, &status);

	// Deallocate memory
	TRACEFREE(rwork);
	TRACEFFTW_FREE(work);
	TRACEFFTW_FREE(Z);
	TRACEFFTW_FREE(A);
}

//void diag_pzheevd(int num_plane_waves, double *H_kinetic,double *H_local,double *eigenvalues)
//{
//	mpi_printf(0, "LAPACK diagonaliser:  PZHEEVD\n");
//	fftw_complex *full_H;
//	fftw_complex *work;
//	double *rwork;
//	int *iwork;
//	int lwork, lrwork, liwork;
//	int i,j;
//	int status;
//	char jobz;
//	char uplo;
//
//	int blacs_size, blacs_rank, blacs_ctxt = 0, blacs_ctxt_root = 0, world_rank, world_size;
//	int dims[2] = {0};
//	int zero = 0;
//	int one = 1;
//
//	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
//	MPI_Comm_size(MPI_COMM_WORLD, &world_size);
//	// First we allocate memory for and construct the full Hamiltonian
//	full_H = (fftw_complex *)TRACEFFTW_MALLOC(num_plane_waves*num_plane_waves*sizeof(fftw_complex));
//
//	construct_full_H(num_plane_waves,H_kinetic,H_local,full_H);
//
//
//	Cblacs_pinfo(&blacs_rank, &blacs_size);
//	//fmpi_printf(0, stderr, "HI!@%d/%d-%d/%d\n", world_rank, world_size, blacs_rank, blacs_size);
//	MPI_Dims_create(blacs_size, 2, dims);
//
//	int nprow, npcol, myprow, mypcol;
//	nprow = dims[0];
//	npcol = dims[1];
//	Cblacs_get(-1, 0, &blacs_ctxt);
//	char cbgi_r = 'R';
//	Cblacs_gridinit(&blacs_ctxt, &cbgi_r, nprow, npcol);
//	Cblacs_gridinit(&blacs_ctxt_root, &cbgi_r, 1, 1);
//	Cblacs_gridinfo(blacs_ctxt, &nprow, &npcol, &myprow, &mypcol);
//
//	//MB = num_plane_waves/nprow;
//	int NB = num_plane_waves/nprow;
//	int MB = NB;//num_plane_waves/nprow;
//	int NLOC_A = numroc_(&num_plane_waves, &NB, &mypcol, &zero, &npcol);
//	int MLOC_A = numroc_(&num_plane_waves, &MB, &myprow, &zero, &nprow);
//
//	int LDA = numroc_(&num_plane_waves, &MB, &myprow, &zero, &nprow);
//	LDA = LDA < 1 ? 1 : LDA;
//	int desc[9]; // apparently DLEN == 9
//	int desc_root[9]; // apparently DLEN == 9
//	descinit_(desc, &num_plane_waves, &num_plane_waves, &MB, &NB, &zero, &zero, &blacs_ctxt, &LDA, &status);
//	if (world_rank == 0) {
//		descinit_(desc_root, &num_plane_waves, &num_plane_waves, &num_plane_waves, &num_plane_waves, &zero, &zero, &blacs_ctxt_root, &num_plane_waves, &status);
//	} else {
//		desc_root[1] = -1;
//	}
//
//	fftw_complex *A = TRACEMALLOC((MLOC_A)*NLOC_A*sizeof(fftw_complex));
//	fftw_complex *Z = TRACEMALLOC((MLOC_A)*NLOC_A*sizeof(fftw_complex));
//
//	// distribute full matrix (from root) to local submatrices
//	pzgemr2d_(&num_plane_waves, &num_plane_waves, full_H, &one, &one, desc_root, A, &one, &one, desc, &blacs_ctxt);
//
//	fftw_complex wsize;
//	double rwsize;
//	int iwsize;
//
//	lwork = lrwork = liwork = -1;
//
//	// Use LAPACK to get eigenvalues and eigenvectors, e.g. the zheev routine
//	// NB H is Hermitian (but not packed)
//	jobz = 'V';
//	uplo = 'U';
//
//	// Workspace query
//	pzheevd_(&jobz, &uplo, &num_plane_waves, A, &one, &one, desc, eigenvalues, Z, &one, &one, desc, &wsize, &lwork, &rwsize, &lrwork, &iwsize, &liwork, &status);
//
//	// Allocate work arrays to rocommended sizes
//	lwork = (int)creal(wsize);
//	lrwork = (int)rwsize;
//	liwork = iwsize;
//
//	work = TRACEFFTW_MALLOC(lwork*sizeof(fftw_complex));
//	rwork = TRACECALLOC(lrwork,sizeof(double));
//	iwork = TRACECALLOC(liwork,sizeof(int));
//
//	// Actual work
//	pzheevd_(&jobz, &uplo, &num_plane_waves, A, &one, &one, desc, eigenvalues, Z, &one, &one, desc, work, &lwork, rwork, &lrwork, iwork, &liwork, &status);
//
//	// Deallocate memory
//	TRACEFREE(iwork);
//	TRACEFREE(rwork);
//	TRACEFFTW_FREE(work);
//	TRACEFFTW_FREE(Z);
//	TRACEFFTW_FREE(A);
//	TRACEFFTW_FREE(full_H);
//}
//
//void diag_pzheevr(int num_plane_waves, int num_states, double *H_kinetic,double *H_local,double *eigenvalues)
//{
//	mpi_printf(0, "LAPACK diagonaliser:  PZHEEVR\n");
//	fftw_complex *full_H;
//	fftw_complex *work;
//	double *rwork;
//	int *iwork;
//	int lwork, lrwork, liwork;
//	int i,j;
//	int status;
//	char jobz;
//	char uplo;
//
//	int blacs_size, blacs_rank, blacs_ctxt = 0, blacs_ctxt_root = 0, world_rank, world_size;
//	int dims[2] = {0};
//	int zero = 0;
//	int one = 1;
//
//	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
//	MPI_Comm_size(MPI_COMM_WORLD, &world_size);
//	// First we allocate memory for and construct the full Hamiltonian
//	full_H = (fftw_complex *)TRACEFFTW_MALLOC(num_plane_waves*num_plane_waves*sizeof(fftw_complex));
//
//	construct_full_H(num_plane_waves,H_kinetic,H_local,full_H);
//
//
//	Cblacs_pinfo(&blacs_rank, &blacs_size);
//	MPI_Dims_create(blacs_size, 2, dims);
//
//	int nprow, npcol, myprow, mypcol;
//	nprow = dims[0];
//	npcol = dims[1];
//	Cblacs_get(-1, 0, &blacs_ctxt);
//	char cbgi_r = 'R';
//	Cblacs_gridinit(&blacs_ctxt, &cbgi_r, nprow, npcol);
//	Cblacs_gridinit(&blacs_ctxt_root, &cbgi_r, 1, 1);
//	Cblacs_gridinfo(blacs_ctxt, &nprow, &npcol, &myprow, &mypcol);
//
//	//MB = num_plane_waves/nprow;
//	int NB = num_plane_waves/nprow;
//	int MB = NB;//num_plane_waves/nprow;
//	int NLOC_A = numroc_(&num_plane_waves, &NB, &mypcol, &zero, &npcol);
//	int MLOC_A = numroc_(&num_plane_waves, &MB, &myprow, &zero, &nprow);
//
//	int LDA = numroc_(&num_plane_waves, &MB, &myprow, &zero, &nprow);
//	LDA = LDA < 1 ? 1 : LDA;
//	int desc[9]; // apparently DLEN == 9
//	int desc_root[9]; // apparently DLEN == 9
//	descinit_(desc, &num_plane_waves, &num_plane_waves, &MB, &NB, &zero, &zero, &blacs_ctxt, &LDA, &status);
//	if (world_rank == 0) {
//		descinit_(desc_root, &num_plane_waves, &num_plane_waves, &num_plane_waves, &num_plane_waves, &zero, &zero, &blacs_ctxt_root, &num_plane_waves, &status);
//	} else {
//		desc_root[1] = -1;
//	}
//
//	fftw_complex *A = TRACEMALLOC((MLOC_A)*NLOC_A*sizeof(fftw_complex));
//	//fftw_complex *Z = NULL; // only used if getting eigenvectors as well
//	fftw_complex *Z = TRACEMALLOC((MLOC_A)*NLOC_A*sizeof(fftw_complex));
//
//	// distribute full matrix (from root) to local submatrices
//	pzgemr2d_(&num_plane_waves, &num_plane_waves, full_H, &one, &one, desc_root, A, &one, &one, desc, &blacs_ctxt);
//
//	lwork = lrwork = liwork = -1;
//
//	//int wsize, rwsize, iwsize;
//	fftw_complex wsize;
//	double rwsize;
//	int iwsize;
//	// Use LAPACK to get eigenvalues and eigenvectors, e.g. the zheev routine
//	// NB H is Hermitian (but not packed)
//	char range = 'A';
//	int VL, VU, IL, IU;
//	int eigenvalues_found, eigenvectors_computed;
//	jobz = 'V';
//	uplo = 'U';
//
//	// Workspace query
//	pzheevr_(&jobz, &range, &uplo, &num_plane_waves, A, &one, &one, desc, 
//			&VL, &VU, &IL, &IU, &eigenvalues_found, &eigenvectors_computed,
//			eigenvalues, Z, &one, &one, desc, &wsize, &lwork, &rwsize, &lrwork, &iwsize, &liwork, &status);
//
//	// allocate recommended sizes;
//	lwork = (int)creal(wsize);
//	lrwork = (int)rwsize;
//	liwork = iwsize;
//
//	work = TRACEFFTW_MALLOC(lwork*sizeof(fftw_complex));
//
//	rwork = TRACECALLOC(lrwork,sizeof(double));
//
//	iwork = TRACECALLOC(liwork,sizeof(int));
//
//	pzheevr_(&jobz, &range, &uplo, &num_plane_waves, A, &one, &one, desc, 
//			&VL, &VU, &IL, &IU, &eigenvalues_found, &eigenvectors_computed,
//			eigenvalues, Z, &one, &one, desc, work, &lwork, rwork, &lrwork, iwork, &liwork, &status);
//
//	//// Deallocate memory
//	TRACEFREE(iwork);
//	TRACEFREE(rwork);
//
//	TRACEFFTW_FREE(work);
//	TRACEFFTW_FREE(Z);
//	TRACEFFTW_FREE(A);
//	TRACEFFTW_FREE(full_H);
//}
