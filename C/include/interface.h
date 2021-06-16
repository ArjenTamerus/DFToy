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
void orthonormalise       (int num_pw, int num_states, fftw_complex *state);
//void orthogonalise        (int num_pw, int num_states, double *state, double *ref_state);
void orthogonalise        (int num_pw, int num_states, fftw_complex *state, fftw_complex *ref_state);
void transform            (int num_pw, int num_states, fftw_complex *state, fftw_complex *transformation);
void diagonalise          (int num_pw, int num_states, fftw_complex *state, fftw_complex *H_state, double *eigenvalues, fftw_complex *rotation);
void precondition         (int num_pw, int num_states, fftw_complex *search_direction, fftw_complex *trial_wvfn, double *H_kinetic);
void line_search          (int num_pw, int num_states, fftw_complex *approx_state,
		double *H_kinetic, double *H_nonlocal, fftw_complex *direction,
		fftw_complex *gradient,  double *eigenvalue, double *energy);
void output_results(int num_pw, int num_states, double *H_local,
		fftw_complex *wvfn);

// LAPACK function prototypes
void   zpotrf_(char *uplo, int *N, fftw_complex *A, int *lda, int *status);
void   ztrtri_(char *uplo, char *jobz, int *N, fftw_complex *A, int *lda, int *status);

// LAPACK ZHEEV*
void   zheev_(char *jobz, char *uplo, int *N, fftw_complex *A, int *ldA, double *w,
		fftw_complex *work, int *lwork, double *rwork, int *status); 

void zheevd_(char *jobz, char *uplo, int *N, fftw_complex *A, int *LDA, double *W,
		fftw_complex *work, int *lwork, double *rwork, int *lrwork, int *iwork,
		int *liwork, int *info);

void zheevr_(char *jobz, char *range, char *uplo, int *N, fftw_complex *A,
		int *ldA, double *vl, double *vu, int *il, int *iu, double *abstol, int *M,
		double *W, fftw_complex *Z, int *ldZ, int *isuppz, fftw_complex *work,
		int *lwork, double *rwork, int *lrwork, int *iwork, int *liwork, int *status);

// ScaLAPACK PZHEEV*
void pzheev_(char *jobz, char *uplo, int *N, fftw_complex *A, int *iA, int *jA, int *descA, double *W, fftw_complex *Z, int *iZ, int *jZ, int *descZ, fftw_complex *work, int *lwork, double *rwork, int *lrwork, int *info);

void pzheevd_(char *jobz, char *uplo, int *N, fftw_complex *A, int *iA, int *jA, int *descA, double *W, fftw_complex *Z, int *iZ, int *jZ, int *descZ, fftw_complex *work, int *lwork, double *rwork, int *lrwork, int *iwork, int *liwork, int *info);

void pzheevr_(char *jobz, char *range, char *uplo, int *N, fftw_complex *A, int *iA, int *jA, int *descA, int *vl, int *vu, int *il, int *iu, int *M, int *nz, double *W, fftw_complex *Z, int *iZ, int *jZ, int *descZ, fftw_complex *work, int *lwork, double *rwork, int *lrwork, int *iwork, int *liwork, int *info);

// ScaLAPACK support routines
void descinit_(int *desc, int *M, int *N, int *MB, int *NB, int *irsrc, int *icsrc, int *ictxt, int *lld, int *info);

int numroc_(int *N, int *NB, int *iproc, int *isrcproc, int *nprocs);

void pzgemr2d_(int *m, int *n, fftw_complex *A, int *ia, int *ja, int *desca, fftw_complex *B, int *ib, int *jb, int *descb, int *ictxt);

// C interface to init_H
void c_init_H(int num_pw, int num_wavevectors, double *H_kinetic, double *H_nonlocal);

// C interface to apply_H
void c_apply_H(int num_pw, int num_states, fftw_complex *state, double *H_kinetic, double *H_nonlocal, fftw_complex *H_state);

// C interface to compute_eigenvalues
void c_compute_eigenvalues(int num_pw,int num_states, double *state, double *H_kinetic, double *H_nonlocal, double *eigenvalues);

// C interface to construct_full_H
void construct_full_H(int num_pw, double *H_kinetic, double *H_nonlocal, fftw_complex *full_H);

// C interface to init_random_seed
void c_init_random_seed();

// C interface to randomise_state()
//void c_randomise_state(int num_pw,int num_states, fftw_complex state[num_states][num_pw]);
void c_randomise_state(int num_pw, int num_states, fftw_complex *state);

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
//void orthogonalise(int num_pw,int num_states, fftw_complex *state, fftw_complex *ref_state) {
//	/* |-------------------------------------------------|
//		 | This subroutine takes a set of states and       |
//		 | orthogonalises them to a set of reference       |
//		 | states.                                         |
//		 |-------------------------------------------------| */
//	char transA;
//	char transB;
//	fftw_complex overlap;
//	int ref_state_offset;
//	int state_offset;
//	int nb1;
//	int nb2;
//	int np;
//
//	int local_ref_state_offset;
//	int local_state_offset;
//
//	/* |---------------------------------------------------------------------|
//		 | You need to:                                                        |
//		 |                                                                     |
//		 | Compute the overlap of ref_state nb1 with state nb2                 |
//		 | (use the generalised "dot product" for complex vectors)             |
//		 |                                                                     |
//		 | Remove the overlapping parts of ref_state nb1 from state nb2        |
//		 |---------------------------------------------------------------------| */
//
//	for (nb2=0;nb2<num_states;nb2++) {
//		int state_offset = nb2*num_pw;
//		for (nb1=0;nb1<num_states;nb1++) {
//			int ref_state_offset = nb1*num_pw;
//
//			overlap = (0+0*I);
//
//			// Calculate overlap
//			// Dot. Prod. = SUM_i(cplx_conj(a)_i*b_i)
//			for (np=0; np < num_pw; np++) {
//				local_ref_state_offset = ref_state_offset+np;
//				local_state_offset = state_offset+np;
//
//				overlap += conj(ref_state[local_ref_state_offset])*state[local_state_offset];
//
//			}
//			
//			//ref_state_offset = 2*nb1*num_pw;
//			// remove overlap from state
//			for (np=0; np < num_pw; np++) {
//				local_ref_state_offset = ref_state_offset+np;
//				local_state_offset = state_offset+np;
//
//				state[local_state_offset] -= overlap*ref_state[local_ref_state_offset];
//			}
//		}
//	}
//
//
//	return;
//
//}
