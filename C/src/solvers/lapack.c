/*
 * Lapack.c
 *
 * Implements LAPACK (ZHEEV*) based diagonalisation.
 *
 */

#include <stdlib.h>
#include <complex.h>
#include <math.h>
#include <time.h>
#include <fftw3.h>
#include <lapacke.h>
#include "parallel.h"
#include "interfaces.h"
#include <diag.h>
#include "trace.h"

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
