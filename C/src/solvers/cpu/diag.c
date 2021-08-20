#include <stdlib.h>
#include <string.h>
#include <complex.h>
#include <math.h>
#include <time.h>
#include <fftw3.h>
#include <lapacke.h>
#include "parallel.h"
#include "interfaces.h"
#include "trace.h"

int zero = 0;
int one = 1;

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
	if(errcode) {
		mpi_error("Error in %p diagonalisation: %d\n", errcode, diag_type);
		MPI_Abort(MPI_COMM_WORLD, errcode);
	}
}

void diag_zheev(fftw_complex *full_H, double *eigenvalues, int num_plane_waves)
{
	int err;

	mpi_printf("Performing exact diagonalisation with ZHEEV...\n");

	err = LAPACKE_zheev(LAPACK_COL_MAJOR, 'V', 'U', num_plane_waves, full_H,
			num_plane_waves, eigenvalues);

	diag_abort_on_error(err, "ZHEEV");
}

void diag_zheevd(fftw_complex *full_H, double *eigenvalues, int num_plane_waves)
{
	int err;

	mpi_printf("Performing exact diagonalisation with ZHEEVD...\n");

	err = LAPACKE_zheevd(LAPACK_COL_MAJOR, 'V', 'U', num_plane_waves, full_H,
			num_plane_waves, eigenvalues);

	diag_abort_on_error(err, "ZHEEVD");
}


void diag_zheevr(fftw_complex *full_H, double *eigenvalues, int num_plane_waves,
		int num_states)
{
	mpi_printf("Performing exact diagonalisation with ZHEEVR...\n");

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

	diag_abort_on_error(err, "ZHEEVR");

}

void diag_pzheev(fftw_complex *full_H, double *eigenvalues, int num_plane_waves)
{
	mpi_printf("Performing exact diagonalisation with PZHEEV...\n");
	fftw_complex *work;
	double *rwork;
	int lwork, lrwork;
	int i,j;
	int status;
	char jobz;
	char uplo;

	int desc[9]; // apparently DLEN == 9
	fftw_complex *A, *Z;

	distribute_matrix_for_diagonaliser(num_plane_waves, &desc[0], full_H, &A, &Z);

	// NOTE: PZHEEV workspace query is broken in Netlib Scalapack <= 2.1
	//
	// So we have to do some ugly maths to manually calculate the work array sizes

	int NB = num_plane_waves/nprow;
	int MB = num_plane_waves/nprow;
	int NP0 = numroc_(&num_plane_waves, &MB, &zero, &zero, &nprow);
	int NQ0 = numroc_(&num_plane_waves, &MB, &zero, &zero, &npcol); // num_plane_waves always >= 2 and >= NB

	// eigenvalues + eigenvectors
	lwork = (NP0+NQ0+NB)*NB + 3*num_plane_waves + num_plane_waves*num_plane_waves;
	work = TRACEFFTW_MALLOC(lwork*sizeof(fftw_complex));

	// values + vectors
	lrwork = 2*(num_plane_waves+num_plane_waves)-2;
	rwork = TRACECALLOC(lrwork,sizeof(double));

	// Use LAPACK to get eigenvalues and eigenvectors, e.g. the zheev routine
	// NB H is Hermitian (but not packed)
	jobz = 'V';
	uplo = 'U';

	/* Can switch to this if ScaLAPACK ever gets fixed
	// Workspace query
	fftw_complex wsize;
	double rwsize;

	lwork = -1; lrwork = -1;
	
	pzheev_(&jobz, &uplo, &num_plane_waves, A, &one, &one, desc, eigenvalues, Z, &one, &one, desc, &wsize, &lwork, &rwsize, &lrwork, &status);

	// Allocate work arrays to recommended size
	lwork = (int)creal(wsize);
	lrwork = (int)rwsize;

	work = TRACEFFTW_MALLOC(lwork*sizeof(fftw_complex));
	rwork = TRACECALLOC(lrwork,sizeof(double));
	*/

	// Actual work
	pzheev_(&jobz, &uplo, &num_plane_waves, A, &one, &one, desc, eigenvalues, Z, &one, &one, desc, work, &lwork, rwork, &lrwork, &status);

	// Deallocate memory
	TRACEFREE(rwork);
	TRACEFFTW_FREE(work);
	TRACEFFTW_FREE(Z);
	TRACEFFTW_FREE(A);
}


void diag_pzheevd(fftw_complex *full_H, double *eigenvalues,
		int num_plane_waves)
{
	mpi_printf("Performing exact diagonalisation with PZHEEVD...\n");
	fftw_complex *work;
	double *rwork;
	int *iwork;
	int lwork, lrwork, liwork;
	int i,j;
	int status;
	char jobz;
	char uplo;

	int desc[9]; // apparently DLEN == 9
	
	fftw_complex *A, *Z;

	distribute_matrix_for_diagonaliser(num_plane_waves, &desc[0], full_H, &A, &Z);

	fftw_complex wsize;
	double rwsize;
	int iwsize;

	lwork = lrwork = liwork = -1;

	// Use LAPACK to get eigenvalues and eigenvectors, e.g. the zheev routine
	// NB H is Hermitian (but not packed)
	jobz = 'V';
	uplo = 'U';

	// Workspace query
	pzheevd_(&jobz, &uplo, &num_plane_waves, A, &one, &one, desc, eigenvalues, Z, &one, &one, desc, &wsize, &lwork, &rwsize, &lrwork, &iwsize, &liwork, &status);

	// Allocate work arrays to rocommended sizes
	lwork = (int)creal(wsize);
	lrwork = (int)rwsize;
	liwork = iwsize;

	work = TRACEFFTW_MALLOC(lwork*sizeof(fftw_complex));
	rwork = TRACECALLOC(lrwork,sizeof(double));
	iwork = TRACECALLOC(liwork,sizeof(int));

	// Actual work
	pzheevd_(&jobz, &uplo, &num_plane_waves, A, &one, &one, desc, eigenvalues, Z, &one, &one, desc, work, &lwork, rwork, &lrwork, iwork, &liwork, &status);

	// Deallocate memory
	TRACEFREE(iwork);
	TRACEFREE(rwork);
	TRACEFFTW_FREE(work);

	TRACEFFTW_FREE(Z);
	TRACEFFTW_FREE(A);
}

void diag_pzheevr(fftw_complex *full_H, double *eigenvalues,
		int num_plane_waves, int num_states)
{
	mpi_printf("Performing exact diagonalisation with PZHEEVR...\n");
	fftw_complex *work;
	double *rwork;
	int *iwork;
	int lwork, lrwork, liwork;
	int i,j;
	int status;
	char jobz;
	char uplo;

	int desc[9]; // apparently DLEN == 9

	fftw_complex *A, *Z;

	distribute_matrix_for_diagonaliser(num_plane_waves, &desc[0], full_H, &A, &Z);

	lwork = lrwork = liwork = -1;

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

	// Workspace query
	pzheevr_(&jobz, &range, &uplo, &num_plane_waves, A, &one, &one, desc, 
			&VL, &VU, &IL, &IU, &eigenvalues_found, &eigenvectors_computed,
			eigenvalues, Z, &one, &one, desc, &wsize, &lwork, &rwsize, &lrwork, &iwsize, &liwork, &status);

	// allocate recommended sizes;
	lwork = (int)creal(wsize);
	lrwork = (int)rwsize;
	liwork = iwsize;

	work = TRACEFFTW_MALLOC(lwork*sizeof(fftw_complex));

	rwork = TRACECALLOC(lrwork,sizeof(double));

	iwork = TRACECALLOC(liwork,sizeof(int));

	pzheevr_(&jobz, &range, &uplo, &num_plane_waves, A, &one, &one, desc, 
			&VL, &VU, &IL, &IU, &eigenvalues_found, &eigenvectors_computed,
			eigenvalues, Z, &one, &one, desc, work, &lwork, rwork, &lrwork, iwork, &liwork, &status);

	//// Deallocate memory
	TRACEFREE(iwork);
	TRACEFREE(rwork);

	TRACEFFTW_FREE(work);
	TRACEFFTW_FREE(Z);
	TRACEFFTW_FREE(A);
}
