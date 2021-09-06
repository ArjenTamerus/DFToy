/*
 * Scalapack.c
 *
 * Implements ScaLAPACK (PZHEEV*) based diagonalisation.
 *
 */

#include <stdlib.h>
#include <string.h>
#include <complex.h>
#include <math.h>
#include <time.h>
#include <fftw3.h>
#include <lapacke.h>
#include "parallel.h"
#include "diag.h"
#include "interfaces.h"
#include "trace.h"

int zero = 0;
int one = 1;

void diag_pzheev(fftw_complex *full_H, double *eigenvalues, int num_plane_waves)
{
	mpi_printf("Performing exact diagonalisation with PZHEEV...\n");
	fftw_complex *work;
	double *rwork;
	int lwork, lrwork;
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

	pzheevd_(&jobz, &uplo, &num_plane_waves, A, &one, &one, desc, eigenvalues, Z, &one, &one, desc, work, &lwork, rwork, &lrwork, iwork, &liwork, &status);
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
