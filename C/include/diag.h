#include <complex.h>
#include <fftw3.h>

// Prototypes for netlib routines

int numroc_(int *n, int *nb, int *iproc, int *isrcproc, int *nprocs);
int descinit_(int *desc, int *m, int *n, int *mb, int *nb, int *isrc,
		int *icsrc, int *ictxt, int *lld, int *info);

// scaLAPACK

int Cblacs_pinfo(int *mypnum, int *nprocs);
int Cblacs_get(int icontxt, int what, int *val);
int Cblacs_gridinit(int *ictxt, char *order, int nprow, int npcol);
int Cblacs_gridinfo(int ictxt, int *nprow, int *npcol, int *myprow,
		int *mypcol);

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
		int *ia, int *ia, int *desca, int *vl, int *vu, int *il, int *iu, int *m,
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

