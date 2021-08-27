#ifndef TC_DIAG_H
#define TC_DIAG_H

#include <complex.h>
#include <fftw3.h>

void pzgemr2d_(int *m, int *n, fftw_complex *A, int *ia, int *ja, int *desca,
		fftw_complex *B, int *ib, int *jb, int *descb, int *ictxt);

void pzheev_(char *jobz, char *uplo, int *n, fftw_complex *A, int *ia, int *ja,
		int *desca, double *W, fftw_complex *Z, int *iz, int *jz, int *descz,
		fftw_complex *work, int *lwork, double *rwork, int *lrwork, int *info);

void pzheevd_(char *jobz, char *uplo, int *n, fftw_complex *A, int *ia, int *ja,
		int *desca, double *W, fftw_complex *Z, int *iz, int *jz, int *descz,
		fftw_complex *work, int *lwork, double *rwork, int *lrwork, int *iwork,
		int *liwork, int *info);

void pzheevr_(char *jobz, char *range, char *uplo, int *n, fftw_complex *A,
		int *ia, int *ja, int *desca, int *vl, int *vu, int *il, int *iu, int *m,
		int *nz, double *W, fftw_complex *Z, int *iz, int *jz, int *descz,
		fftw_complex *work, int *lwork, double *rwork, int *lrwork, int *iwork,
		int *liwork, int *info);

// solvers

void diag_zheev(fftw_complex *full_H, double *eigenvalues, int num_plane_waves);
void diag_zheevd(fftw_complex *full_H, double *eigenvalues, int num_plane_waves);
void diag_zheevr(fftw_complex *full_H, double *eigenvalues, int num_plane_waves,
		int num_states);

void diag_pzheev(fftw_complex *full_H, double *eigenvalues, int num_plane_waves);
void diag_pzheevd(fftw_complex *full_H, double *eigenvalues,
		int num_plane_waves);
void diag_pzheevr(fftw_complex *full_H, double *eigenvalues,
		int num_plane_waves, int num_states);

void diag_elpa(fftw_complex *full_H, double *eigenvalues, int num_plane_waves);

#endif
