#include <complex.h>
#include <fftw3.h>

// Prototypes for netlib routines

int numroc_(int *a, int *b, int *c, int *d, int *e);
int descinit_(int *a, int *b, int *c, int *d, int *e, int *f, int *g, int *h,
		int *i, int *j);

// scaLAPACK

int Cblacs_pinfo(int *blacs_rank, int *blacs_size);
int Cblacs_get(int a, int b, int *blacs_ctxt);
int Cblacs_gridinit(int *blacs_ctxt, char *a, int b, int c);
int Cblacs_gridinfo(int blacs_ctxt, int *nprow, int *npcol, int *myprow,
		int *mypcol);

void pzgemr2d_(int *a, int *b, fftw_complex *d, int *e, int *f, int *g,
		fftw_complex *h, int *i, int *j, int *k, int *blacs_ctxt);
void pzheev_(char *a, char *b, int *c, fftw_complex *A, int *e, int *f, int *g,
		double *h, fftw_complex *Z, int *i, int *j, int *k, fftw_complex *work,
		int *lwork, double *rwork, int *lrwork, int *status);

void pzheevd_(char *a, char *b, int *c, fftw_complex *A, int *e, int *f, int *g,
		double *h, fftw_complex *Z, int *i, int *j, int *k, fftw_complex *work,
		int *lwork, double *rwork, int *lrwork, int *iwork, int *liwork,
		int *status);

void pzheevr_(char *jobz, char *range, char *uplo, int *c, fftw_complex *A,
		int *e, int *f, int *desc, int *VL, int *VU, int *IL, int *IU, int *g,
		int *g2, double *h, fftw_complex *Z, int *i, int *j, int *k,
		fftw_complex *work, int *lwork, double *rwork, int *lrwork, int *iwork,
		int *liwork, int *status);

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

