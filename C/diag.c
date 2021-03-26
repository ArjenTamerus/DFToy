#include <stdlib.h>
#include <string.h>
#include <complex.h>
#include <math.h>
#include <time.h>
#include <mpi.h>
#include <fftw3.h>
#include "interface.h"
#include "trace.h"

#include <PBblacs.h>

int get_diag_mode() {
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
	}

	return mode;
}

void diag_zheev(int num_pw, double *H_kinetic,double *H_local,double *full_eigenvalue)
{
	printf("LAPACK diagonaliser:  ZHEEV\n");
	fftw_complex *full_H;
	fftw_complex *work;
	double *rwork;
	int *iwork;
	int lwork, lrwork, liwork;
	int i;
	int status;
	char jobz;
	char uplo;

	// First we allocate memory for and construct the full Hamiltonian
	full_H = (fftw_complex *)TRACEMALLOC(num_pw*num_pw*sizeof(fftw_complex));

	construct_full_H(num_pw,H_kinetic,H_local,full_H);

	lwork = (2*num_pw)-1;
	work = TRACEFFTW_MALLOC(lwork*sizeof(fftw_complex));

	lrwork = (3*num_pw)-2;
	rwork = TRACECALLOC(lrwork,sizeof(double));

	// Use LAPACK to get eigenvalues and eigenvectors, e.g. the zheev routine
	// NB H is Hermitian (but not packed)
	jobz = 'V';
	uplo = 'U';
	zheev_(&jobz, &uplo, &num_pw, full_H, &num_pw, full_eigenvalue, work, &lwork, rwork, &status);

	// Deallocate memory
	TRACEFREE(rwork);
	TRACEFFTW_FREE(work);
	TRACEFFTW_FREE(full_H);
}

void diag_zheevd(int num_pw, double *H_kinetic,double *H_local,double *full_eigenvalue)
{
	printf("LAPACK diagonaliser:  ZHEEVD\n");
	fftw_complex *full_H;
	fftw_complex *work;
	double *rwork;
	int *iwork;
	int lwork, lrwork, liwork;
	int i;
	int status;
	char jobz;
	char uplo;

	// First we allocate memory for and construct the full Hamiltonian
	full_H = (fftw_complex *)TRACEMALLOC(num_pw*num_pw*sizeof(fftw_complex));

	construct_full_H(num_pw,H_kinetic,H_local,full_H);

	lwork = 2*num_pw+num_pw*num_pw;
	work = TRACEFFTW_MALLOC(lwork*sizeof(fftw_complex));

	lrwork = 1+5*num_pw+2*num_pw*num_pw;
	rwork = TRACECALLOC(lrwork,sizeof(double));

	liwork = 3+5*num_pw;
	iwork = TRACECALLOC(liwork,sizeof(int));

	// Use LAPACK to get eigenvalues and eigenvectors, e.g. the zheev routine
	// NB H is Hermitian (but not packed)
	jobz = 'V';
	uplo = 'U';

	zheevd_( &jobz, &uplo, &num_pw, full_H, &num_pw, full_eigenvalue,
			work, &lwork, rwork, &lrwork,
			iwork, &liwork, &status);

	// Deallocate memory
	TRACEFREE(rwork);
	TRACEFFTW_FREE(work);
	TRACEFREE(iwork);
	TRACEFFTW_FREE(full_H);
}

void diag_zheevr(int num_pw, int num_states, double *H_kinetic,double *H_local,double *full_eigenvalue)
{
	printf("LAPACK diagonaliser:  ZHEEVR\n");
	fftw_complex *full_H;
	fftw_complex *work;
	double *rwork;
	int *iwork;
	int lwork, lrwork, liwork;
	int i;
	int status;
	char jobz;
	char uplo;

	// First we allocate memory for and construct the full Hamiltonian
	full_H = (fftw_complex *)TRACEMALLOC(num_pw*num_pw*sizeof(fftw_complex));

	construct_full_H(num_pw,H_kinetic,H_local,full_H);

	lwork = 2*num_pw; //TODO: lwork >= (NB+1*N) see docs for NB
	work = TRACEFFTW_MALLOC(lwork*sizeof(fftw_complex));

	lrwork = 24*num_pw;
	rwork = TRACECALLOC(lrwork,sizeof(double));

	liwork = 10*num_pw;
	iwork = TRACECALLOC(liwork,sizeof(int));

	// Use LAPACK to get eigenvalues and eigenvectors, e.g. the zheev routine
	// NB H is Hermitian (but not packed)
	jobz = 'V';
	uplo = 'U';

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
	//int *isuppz = TRACECALLOC(num_states, sizeof(int));

	// zheevr-specific arrays and descriptors
	int ldZ = num_pw; // Some kind of aligment glitch?
	fftw_complex *z_work = TRACEFFTW_MALLOC(ldZ*num_states*sizeof(fftw_complex));

	zheevr_(&jobz, &range, &uplo, &num_pw, full_H, &num_pw,
			&VL, &VU, &IL, &IU, &abstol, &eigenvals_found, full_eigenvalue, z_work, &ldZ, isuppz,
			work, &lwork, rwork, &lrwork, iwork, &liwork,
			&status);

	// Deallocate memory
	TRACEFREE(rwork);
	TRACEFFTW_FREE(work);
	TRACEFREE(iwork);
	TRACEFFTW_FREE(z_work);
	TRACEFREE(isuppz);
	TRACEFFTW_FREE(full_H);
}

void diag_pzheev(int num_pw, double *H_kinetic,double *H_local,double *full_eigenvalue)
{
	printf("LAPACK diagonaliser:  PZHEEV\n");
	fftw_complex *full_H;
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
	// First we allocate memory for and construct the full Hamiltonian
	full_H = (fftw_complex *)TRACEMALLOC(num_pw*num_pw*sizeof(fftw_complex));

	construct_full_H(num_pw,H_kinetic,H_local,full_H);


	Cblacs_pinfo(&blacs_rank, &blacs_size);
	//fprintf(stderr, "HI!@%d/%d-%d/%d\n", world_rank, world_size, blacs_rank, blacs_size);
	MPI_Dims_create(blacs_size, 2, dims);

	int nprow, npcol, myprow, mypcol;
	nprow = dims[0];
	npcol = dims[1];
	Cblacs_get(-1, 0, &blacs_ctxt);
	char cbgi_r = 'R';
	Cblacs_gridinit(&blacs_ctxt, &cbgi_r, nprow, npcol);
	Cblacs_gridinit(&blacs_ctxt_root, &cbgi_r, 1, 1);
	Cblacs_gridinfo(blacs_ctxt, &nprow, &npcol, &myprow, &mypcol);
	fprintf(stderr, "HI!@%d/%d-%d/%d\n", myprow, nprow, mypcol, npcol);

	//MB = num_pw/nprow;
	int NB = num_pw/nprow;
	int MB = NB;//num_pw/nprow;
	int NLOC_A = numroc_(&num_pw, &NB, &mypcol, &zero, &npcol);
	int MLOC_A = numroc_(&num_pw, &MB, &myprow, &zero, &nprow);
	printf("[%d] LOCS: %d %d \n", world_rank, MLOC_A, NLOC_A);

	int LDA = numroc_(&num_pw, &MB, &myprow, &zero, &nprow);
	LDA = LDA < 1 ? 1 : LDA;
	int desc[9]; // apparently DLEN == 9
	int desc_root[9]; // apparently DLEN == 9
	descinit_(desc, &num_pw, &num_pw, &MB, &NB, &zero, &zero, &blacs_ctxt, &LDA, &status);
	if (world_rank == 0) {
		descinit_(desc_root, &num_pw, &num_pw, &num_pw, &num_pw, &zero, &zero, &blacs_ctxt_root, &num_pw, &status);
	} else {
		desc_root[1] = -1;
	}

	fftw_complex *A = TRACEMALLOC((MLOC_A)*NLOC_A*sizeof(fftw_complex));
	//fftw_complex *Z = NULL; // only used if getting eigenvectors as well
	fftw_complex *Z = TRACEMALLOC((MLOC_A)*NLOC_A*sizeof(fftw_complex));

	// distribute full matrix (from root) to local submatrices
	pzgemr2d_(&num_pw, &num_pw, full_H, &one, &one, desc_root, A, &one, &one, desc, &blacs_ctxt);

	//// LWORK
	int NP0 = numroc_(&num_pw, &MB, &zero, &zero, &nprow);
	int NQ0 = numroc_(&num_pw, &MB, &zero, &zero, &npcol); // num_pw always >= 2 and >= NB

	//// eigenvalues only
	////int max = NB*(NP0+1) > 3 ? NB*(NP0+1): 3;
	////lwork = max + 3*num_pw;
	//// eigenvalues + eigenvectors
	lwork = (NP0+NQ0+NB)*NB + 3*num_pw + num_pw*num_pw;
	work = TRACEFFTW_MALLOC(lwork*sizeof(fftw_complex));

	//// values
	////lrwork = 2*num_pw;
	//// values + vectors
	lrwork = 2*(num_pw+num_pw)-2;
	rwork = TRACECALLOC(lrwork,sizeof(double));
	printf("Pre-query: %d\t%d\n", lwork, lrwork);

	//lwork = -1; lrwork = -1;
	//fftw_complex wsize;
	//double rwsize;
	// Use LAPACK to get eigenvalues and eigenvectors, e.g. the zheev routine
	// NB H is Hermitian (but not packed)
	jobz = 'V';
	uplo = 'U';

	//pzheev_(&jobz, &uplo, &num_pw, A, &one, &one, desc, full_eigenvalue, Z, &one, &one, desc, &wsize, &lwork, &rwsize, &lrwork, &status);
	//printf("Work Query results A: %f+%fi\t%f\n", creal(wsize), cimag(wsize), rwsize);

	//lwork = (int)creal(wsize);
	//lrwork = (int)rwsize;

	////lwork = wsize;
	//work = TRACEFFTW_MALLOC(lwork*sizeof(fftw_complex));

	//////lrwork = rwsize;
	//rwork = TRACECALLOC(lrwork,sizeof(double));
	pzheev_(&jobz, &uplo, &num_pw, A, &one, &one, desc, full_eigenvalue, Z, &one, &one, desc, work, &lwork, rwork, &lrwork, &status);

	//// Deallocate memory
	TRACEFREE(rwork);
	TRACEFFTW_FREE(work);
	TRACEFFTW_FREE(Z);
	TRACEFFTW_FREE(A);
	TRACEFFTW_FREE(full_H);
}

void diag_pzheevd(int num_pw, double *H_kinetic,double *H_local,double *full_eigenvalue)
{
	printf("LAPACK diagonaliser:  PZHEEVD\n");
	fftw_complex *full_H;
	fftw_complex *work;
	double *rwork;
	int *iwork;
	int lwork, lrwork, liwork;
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
	// First we allocate memory for and construct the full Hamiltonian
	full_H = (fftw_complex *)TRACEMALLOC(num_pw*num_pw*sizeof(fftw_complex));

	construct_full_H(num_pw,H_kinetic,H_local,full_H);


	Cblacs_pinfo(&blacs_rank, &blacs_size);
	//fprintf(stderr, "HI!@%d/%d-%d/%d\n", world_rank, world_size, blacs_rank, blacs_size);
	MPI_Dims_create(blacs_size, 2, dims);

	int nprow, npcol, myprow, mypcol;
	nprow = dims[0];
	npcol = dims[1];
	Cblacs_get(-1, 0, &blacs_ctxt);
	char cbgi_r = 'R';
	Cblacs_gridinit(&blacs_ctxt, &cbgi_r, nprow, npcol);
	Cblacs_gridinit(&blacs_ctxt_root, &cbgi_r, 1, 1);
	Cblacs_gridinfo(blacs_ctxt, &nprow, &npcol, &myprow, &mypcol);
	fprintf(stderr, "HI!@%d/%d-%d/%d\n", myprow, nprow, mypcol, npcol);

	//MB = num_pw/nprow;
	int NB = num_pw/nprow;
	int MB = NB;//num_pw/nprow;
	int NLOC_A = numroc_(&num_pw, &NB, &mypcol, &zero, &npcol);
	int MLOC_A = numroc_(&num_pw, &MB, &myprow, &zero, &nprow);
	printf("[%d] LOCS: %d %d \n", world_rank, MLOC_A, NLOC_A);

	int LDA = numroc_(&num_pw, &MB, &myprow, &zero, &nprow);
	LDA = LDA < 1 ? 1 : LDA;
	int desc[9]; // apparently DLEN == 9
	int desc_root[9]; // apparently DLEN == 9
	descinit_(desc, &num_pw, &num_pw, &MB, &NB, &zero, &zero, &blacs_ctxt, &LDA, &status);
	if (world_rank == 0) {
		descinit_(desc_root, &num_pw, &num_pw, &num_pw, &num_pw, &zero, &zero, &blacs_ctxt_root, &num_pw, &status);
	} else {
		desc_root[1] = -1;
	}

	fftw_complex *A = TRACEMALLOC((MLOC_A)*NLOC_A*sizeof(fftw_complex));
	//fftw_complex *Z = NULL; // only used if getting eigenvectors as well
	fftw_complex *Z = TRACEMALLOC((MLOC_A)*NLOC_A*sizeof(fftw_complex));

	// distribute full matrix (from root) to local submatrices
	pzgemr2d_(&num_pw, &num_pw, full_H, &one, &one, desc_root, A, &one, &one, desc, &blacs_ctxt);

	// LWORK
	int NP0 = numroc_(&num_pw, &MB, &zero, &zero, &nprow);
	int MQ0 = numroc_(&num_pw, &MB, &zero, &zero, &npcol); // num_pw always >= 2 and >= NB

	lwork = num_pw + (NP0+MQ0+NB)*NB; 
	//work = TRACEFFTW_MALLOC(lwork*sizeof(fftw_complex));

	int NP = numroc_(&num_pw, &NB, &myprow, &zero, &nprow);
	int NQ = numroc_(&num_pw, &NB, &mypcol, &zero, &npcol);

	lrwork = 1 + 9*num_pw + 3*NP*NQ;
	//rwork = TRACECALLOC(lrwork,sizeof(double));

	liwork = 7*num_pw + 8*npcol + 2;
	//iwork = TRACECALLOC(liwork,sizeof(int));
	printf("Pre-query: %d\t%d\n", lwork, lrwork);
	lwork = lrwork = liwork = -1;
	fftw_complex wsize;
	double rwsize;
	int iwsize;

	// Use LAPACK to get eigenvalues and eigenvectors, e.g. the zheev routine
	// NB H is Hermitian (but not packed)
	jobz = 'V';
	uplo = 'U';
	pzheevd_(&jobz, &uplo, &num_pw, A, &one, &one, desc, full_eigenvalue, Z, &one, &one, desc, &wsize, &lwork, &rwsize, &lrwork, &iwsize, &liwork, &status);
	printf("Work Query results A: %f+%fi\t%f\t%d\n", creal(wsize), cimag(wsize), rwsize, iwsize);

	lwork = (int)creal(wsize);
	lrwork = (int)rwsize;
	liwork = iwsize;

	//lwork = wsize;
	work = TRACEFFTW_MALLOC(lwork*sizeof(fftw_complex));

	////lrwork = rwsize;
	rwork = TRACECALLOC(lrwork,sizeof(double));

	////liwork = iwsize;
	iwork = TRACECALLOC(liwork,sizeof(int));
	pzheevd_(&jobz, &uplo, &num_pw, A, &one, &one, desc, full_eigenvalue, Z, &one, &one, desc, work, &lwork, rwork, &lrwork, iwork, &liwork, &status);
	//// Deallocate memory
	TRACEFREE(iwork);
	TRACEFREE(rwork);
	TRACEFFTW_FREE(work);
	TRACEFFTW_FREE(Z);
	TRACEFFTW_FREE(A);
	TRACEFFTW_FREE(full_H);
}

void diag_pzheevr(int num_pw, int num_states, double *H_kinetic,double *H_local,double *full_eigenvalue)
{
	printf("LAPACK diagonaliser:  PZHEEVR\n");
	fftw_complex *full_H;
	fftw_complex *work;
	double *rwork;
	int *iwork;
	int lwork, lrwork, liwork;
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
	// First we allocate memory for and construct the full Hamiltonian
	full_H = (fftw_complex *)TRACEMALLOC(num_pw*num_pw*sizeof(fftw_complex));

	construct_full_H(num_pw,H_kinetic,H_local,full_H);


	Cblacs_pinfo(&blacs_rank, &blacs_size);
	//fprintf(stderr, "HI!@%d/%d-%d/%d\n", world_rank, world_size, blacs_rank, blacs_size);
	MPI_Dims_create(blacs_size, 2, dims);

	int nprow, npcol, myprow, mypcol;
	nprow = dims[0];
	npcol = dims[1];
	Cblacs_get(-1, 0, &blacs_ctxt);
	char cbgi_r = 'R';
	Cblacs_gridinit(&blacs_ctxt, &cbgi_r, nprow, npcol);
	Cblacs_gridinit(&blacs_ctxt_root, &cbgi_r, 1, 1);
	Cblacs_gridinfo(blacs_ctxt, &nprow, &npcol, &myprow, &mypcol);
	fprintf(stderr, "HI!@%d/%d-%d/%d\n", myprow, nprow, mypcol, npcol);

	//MB = num_pw/nprow;
	int NB = num_pw/nprow;
	int MB = NB;//num_pw/nprow;
	int NLOC_A = numroc_(&num_pw, &NB, &mypcol, &zero, &npcol);
	int MLOC_A = numroc_(&num_pw, &MB, &myprow, &zero, &nprow);
	printf("[%d] LOCS: %d %d \n", world_rank, MLOC_A, NLOC_A);

	int LDA = numroc_(&num_pw, &MB, &myprow, &zero, &nprow);
	LDA = LDA < 1 ? 1 : LDA;
	int desc[9]; // apparently DLEN == 9
	int desc_root[9]; // apparently DLEN == 9
	descinit_(desc, &num_pw, &num_pw, &MB, &NB, &zero, &zero, &blacs_ctxt, &LDA, &status);
	if (world_rank == 0) {
		descinit_(desc_root, &num_pw, &num_pw, &num_pw, &num_pw, &zero, &zero, &blacs_ctxt_root, &num_pw, &status);
	} else {
		desc_root[1] = -1;
	}

	fftw_complex *A = TRACEMALLOC((MLOC_A)*NLOC_A*sizeof(fftw_complex));
	//fftw_complex *Z = NULL; // only used if getting eigenvectors as well
	fftw_complex *Z = TRACEMALLOC((MLOC_A)*NLOC_A*sizeof(fftw_complex));

	// distribute full matrix (from root) to local submatrices
	pzgemr2d_(&num_pw, &num_pw, full_H, &one, &one, desc_root, A, &one, &one, desc, &blacs_ctxt);

	lwork = -1;
	lrwork = -1;
	liwork = -1;

	//int wsize, rwsize, iwsize;
	fftw_complex wsize;
	double rwsize;
	int iwsize;
	// Use LAPACK to get eigenvalues and eigenvectors, e.g. the zheev routine
	// NB H is Hermitian (but not packed)
	char range = 'A';
	int VL, VU, IL, IU;
	int eigenvalues_found, eigenvectors_computed;
	jobz = 'V';
	uplo = 'U';
	pzheevr_(&jobz, &range, &uplo, &num_pw, A, &one, &one, desc, 
			&VL, &VU, &IL, &IU, &eigenvalues_found, &eigenvectors_computed,
			full_eigenvalue, Z, &one, &one, desc, &wsize, &lwork, &rwsize, &lrwork, &iwsize, &liwork, &status);
	printf("Work Query results A: %f+%fi\t%f\t%d\n", creal(wsize), cimag(wsize), rwsize, iwsize);

	lwork = (int)creal(wsize);
	lrwork = (int)rwsize;
	liwork = iwsize;

	//lwork = wsize;
	work = TRACEFFTW_MALLOC(lwork*sizeof(fftw_complex));

	////lrwork = rwsize;
	rwork = TRACECALLOC(lrwork,sizeof(double));

	////liwork = iwsize;
	iwork = TRACECALLOC(liwork,sizeof(int));

	pzheevr_(&jobz, &range, &uplo, &num_pw, A, &one, &one, desc, 
			&VL, &VU, &IL, &IU, &eigenvalues_found, &eigenvectors_computed,
			full_eigenvalue, Z, &one, &one, desc, work, &lwork, rwork, &lrwork, iwork, &liwork, &status);

	//// Deallocate memory
	//TRACEFREE(iwork);
	//TRACEFREE(rwork);
	//TRACEFFTW_FREE(work);
	TRACEFFTW_FREE(Z);
	TRACEFFTW_FREE(A);
	TRACEFFTW_FREE(full_H);
}
