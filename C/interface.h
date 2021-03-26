#ifndef INTERFACE_H__
#define INTERFACE_H__
// Global variables
// (first two arrays should _not_ be here permanently, most likely)
extern double *H_kinetic;        // the kinetic energy operator
extern double *H_local;          // the local potential operator

extern double trial_step; // step for line search
extern unsigned int seed;       // random number seed

// Function prototypes
void exact_diagonalisation(int num_pw, int num_states, double *H_kinetic, double *H_nonlocal, double *full_eigenvalue, int diag_mode);
void orthonormalise       (int num_pw, int num_states, double *state);
void orthogonalise        (int num_pw, int num_states, double *state, double *ref_state);
void transform            (int num_pw, int num_states, double *state, double *transformation);
void diagonalise          (int num_pw, int num_states, double *state, double *H_state, double *eigenvalues, double *rotation);
void precondition         (int num_pw, int num_states, double *search_direction, double *trial_wvfn, double *H_kinetic);
void line_search          (int num_pw, int num_states, double *approx_state,
		double *H_kinetic, double *H_nonlocal, double *direction,
		double *gradient,  double *eigenvalue, double *energy);
void output_results(int num_pw, int num_states, double* H_local,
		double* wvfn);

// LAPACK function prototypes
void   zheev_(char *jobz, char *uplo, int *N, fftw_complex *A, int *ldA, double *w,
		fftw_complex *cwork, int *lwork, double *work, int *status); 

void zheevd_(char *jobz, char *uplo, int *N, fftw_complex *A, int *LDA, double *W,
		fftw_complex *work, int *lwork, double *rwork, int *lrwork, int *iwork,
		int *liwork, int *info);

void zheevr_(char *jobz, char *range, char *uplo, int *N, fftw_complex *A,
		int *ldA, double *vl, double *vu, int *il, int *iu, double *abstol, int *M,
		double *W, fftw_complex *Z, int *ldZ, int *isuppz, fftw_complex *work,
		int *lwork, double *rwork, int *lrwork, int *iwork, int *liwork, int *status);

void   zpotrf_(char *uplo, int *N,double *A,int *lda,int *status);
void   ztrtri_(char *uplo, char *jobz, int *N, double *A,int *lda, int *status);

// C interface to init_H
void c_init_H(int num_pw, double *H_kinetic, double *H_nonlocal);

// C interface to apply_H
void c_apply_H(int num_pw, int num_states, double *state, double *H_kinetic, double *H_nonlocal, double *H_state);

// C interface to compute_eigenvalues
void c_compute_eigenvalues(int num_pw,int num_states, double *state, double *H_kinetic, double *H_nonlocal, double *eigenvalues);

// C interface to construct_full_H
void construct_full_H(int num_pw, double *H_kinetic, double *H_nonlocal, fftw_complex *full_H);

// C interface to init_random_seed
void c_init_random_seed();

// C interface to randomise_state()
void c_randomise_state(int num_pw,int num_states, fftw_complex state[num_states][num_pw]);

// C interface to line_search
//double c_line_search(int num_pw, int num_states, double *approx_state, double *H_kinetic, double *H_nonlocal, double *direction, double energy);


int get_diag_mode();
void diag_zheev(int num_pw, double *H_kinetic,double *H_local,double *full_eigenvalue);
void diag_zheevd(int num_pw, double *H_kinetic,double *H_local,double *full_eigenvalue);
void diag_zheevr(int num_pw, int num_states, double *H_kinetic,double *H_local,double *full_eigenvalue);
void diag_pzheev(int num_pw, double *H_kinetic,double *H_local,double *full_eigenvalue);
void diag_pzheevd(int num_pw, double *H_kinetic,double *H_local,double *full_eigenvalue);
void diag_pzheevr(int num_pw, int num_states, double *H_kinetic,double *H_local,double *full_eigenvalue);
#endif
