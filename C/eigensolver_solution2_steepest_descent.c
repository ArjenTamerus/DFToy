/* |-------------------------------------------------|
   | This program is designed to illustrate the use  |
   | of numerical methods and optimised software     |
   | libraries to solve large Hermitian eigenvalue   |
   | problems where only the lowest few eigenstates  |
   | are required. Such problems arise in many       |
   | electronic structure calculations, where we     |
   | solve a large Schroedinger equation for just    |
   | enough states to accommodate the number of      |
   | electrons.                                      |
   |                                                 |
   | In this program we will use the special case    |
   | where the Hamiltonian is real and symmetric.    |
   | This is a good approximation for large systems. |
   |-------------------------------------------------|
   | C version by Phil Hasnip (University of York)   |
   | and Dave Quigley (University of Warwick)        |
   | with help from Peter Hill (University of York)  |
   |-------------------------------------------------|
   | Version 0.11, last modified 10th Sept. 2020     |
   |-------------------------------------------------| */

/* |---------------------------------------------------------|
   | Notes specific to this C program                        |
   |                                                         |
   | 1. Complex arguments to library subroutines             |
   |                                                         |
   | This program calls some subroutines in the              |
   | libhamiltonian library, which is written in Fortran     |
   | and uses Fortran's intrinsic complex datatype. In this  |
   | program we have assumed that we can treat such complex  |
   | variables as double-precision variables of twice the    |
   | length, and that complex arrays are ordered such that   |
   | the memory layout of an array A with elements           |
   | A_1, A_2 etc. is:                                       |
   |                                                         |
   |   real(A_1) imag(A_1) real(A_2) imag(A_2) ...           |
   |                                                         |
   | 2. 2D array arguments to library subroutines            |
   |                                                         |
   | Some library subroutines also uses 2D arrays in places, |
   | which are here malloc'd as 1D arrays and manually       |
   | indexed in an identical way to Fortran                  |
   | (i.e. column-major order).                              |
   |                                                         |
   | 3. Complex algebra                                      |
   |                                                         |
   | Special care needs to be taken when computing           |
   | dot-products of complex vectors, for example            |
   | wavefunctions. If we have 2 complex vectors C and D,    |
   | then the dot-product is:                                |
   |                                                         |
   |     C . D = conjg(C) * D                                |
   |                                                         |
   | Using complex elements, this gives                      |
   |                                                         |
   | C . D = sum_n conjg(C_n) * D_n                          |
   |       = sum_n (Re(C_n) - Im(C_n)i)*(Re(D_n) + Im(D_n)i  |
   |       = sum_n Re(C_n)Re(D_n) + Im(C_n)Im(D_n)           |
   |             +  (Re(C_n)Im(D_n) - Im(C_n)Re(D_n)) i      |
   |                                                         |
   | which, rewritten in terms of our double-precision,      |
   | double-length arrays is:                                |
   |                                                         |
   | C . D = sum_n Re(C_2n)Re(D_2n) + Im(C_2n+1)Im(D_2n+1)   |
   |          +  (Re(C_2n)Im(D_2n+1) - Im(C_2n+1)Re(D_2n)) i |
   |---------------------------------------------------------| */


#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <hamiltonian.h>
#include <fftw3.h>

// Global variables
double trial_step = 0.4; // step for line search
unsigned int seed;       // random number seed

// Function prototypes
void exact_diagonalisation(int num_pw, double *H_kinetic, double *H_nonlocal, double *full_eigenvalue);
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
void   zheev_(char *jobz, char *uplo, int *N, double *A, int *ldA, double *w,
	      double *cwork, int *lwork, double *work, int *status); 
void   zpotrf_(char *uplo, int *N,double *A,int *lda,int *status);
void   ztrtri_(char *uplo, char *jobz, int *N, double *A,int *lda, int *status);

// Main program unit
int main() {

/* ----------------------------------------------------
   | Declare variables                                |
   ---------------------------------------------------- */

  double *H_kinetic;        // the kinetic energy operator
  double *H_local;          // the local potential operator
  int     num_wavevectors;  // no. wavevectors in the basis
  int     num_pw;           // no. plane waves
  int     num_states;       // no. eigenstates req'd
  
  double *trial_wvfn;       // current best guess at wvfn
  double *gradient;         // gradient of energy
  double *search_direction; // direction to search along
  double *prev_search_direction; // last direction searched along (for CG)

  double *eigenvalue;       // best guess at eigenvalues
  double *products;         // dot products of current states

  double *full_eigenvalue; // LAPACK computed eigenvalues
  
  double exact_energy;      // total energy from exact eigenvalues
  double energy;            // current total energy
  double prev_energy ;      // energy at last cycle
  double energy_tol;        // convergence tolerance on energy
  
  int max_iter;             // maximum number of iterations
  int iter,i,nb;            // loop counters

  // extra variables for conjugate gradients
  double cg_gamma;
  double cg_beta;
  double cg_beta_old;
  int    reset_sd = 0; // 0 means reset to steepest descent

  // rotation returned by diagonalise
  double *rotation;

  // Timing variables
  clock_t init_cpu_time,curr_cpu_time,exact_cpu_time,iter_cpu_time;
 
/* ---------------------
   | Initialise system |
   --------------------*/

  // No. nonzero wavevectors "G" in our wavefunction expansion
  num_wavevectors = 100;

  // No. plane-waves in our wavefunction expansion. One plane-wave has
  // wavevector 0, and for all the others there are plane-waves at +/- G
  num_pw = 2*num_wavevectors+1;
  
  // No. eigenstates to compute
  num_states = 1;

  // Catch any nonsensical combinations of parameters
  if (num_states>=num_pw) {
    printf("'Error, num_states must be less than num_pw\n'");
    exit(EXIT_FAILURE);
  } 

  // Set tolerance on the eigenvalue sum when using an iterative search. 
  // The iterative search will stop when the change in the eigenvalue sum
  // per iteration is less than this tolerance.
  energy_tol = 1.0e-10;

  // Initialise random number generator
  c_init_random_seed();

  printf("Initialising Hamiltonian...\n");
  
  // Initialise and build the Hamiltonian, comprising two terms: the kinetic energy,
  // which is a diagonal matrix in the plane-wave basis (Fourier space); and the local
  // potential energy, which is a diagonal matrix in real space (direct space).
  //
  // The Hamiltonian is built inside libhamiltonian by populating these two arrays.
  // NB these arrays are *real* (not complex)
  H_kinetic  = (double *)malloc(num_pw*sizeof(double));  
  H_local = (double *)malloc(num_pw*sizeof(double));
  c_init_H(num_pw,H_kinetic,H_local);

/* ----------------------------------------------------
   | Perform full diagonalisation using LAPACK.       |
   | You will need to complete the function called    |
   ---------------------------------------------------- */
  printf("Starting full diagonalisation...\n\n");

  init_cpu_time    = clock();
  full_eigenvalue = (double *)malloc(num_pw*sizeof(double));
  exact_diagonalisation(num_pw,H_kinetic,H_local,full_eigenvalue);
  curr_cpu_time    = clock();

  exact_cpu_time = curr_cpu_time-init_cpu_time;

  printf(" State         Eigenvalue\n");
  for (nb=0;nb<num_states;nb++) {
    printf("     %d % #19.10g\n",1+nb,full_eigenvalue[nb]);
  }

  // Energy is the sum of the eigenvalues of the occupied states
  exact_energy = 0.0;
  for (nb=0;nb<num_states;nb++) { exact_energy += full_eigenvalue[nb]; }
  printf("Ground state energy: % #16.10g\n",exact_energy);

  printf("Full diagonalisation took %f secs\n\n",(double)exact_cpu_time/(double)CLOCKS_PER_SEC);
 
  // Allocate memory for iterative eigenvector search. Each of the following
  // are stored in column-major (Fortran) order. Each column contains the 
  // plane wave co-efficients for a single particle wavefunction and there
  // are num_states particles.
  //
  // NB these are *complex* so need 2 "doubles" per element (one real, one imaginary)
  trial_wvfn            = (double *)malloc(2*num_pw*num_states*sizeof(double));
  gradient              = (double *)malloc(2*num_pw*num_states*sizeof(double));
  search_direction      = (double *)malloc(2*num_pw*num_states*sizeof(double));
  prev_search_direction = (double *)malloc(2*num_pw*num_states*sizeof(double));

  // We have num_states eigenvalue estimates (real) and products (complex)
  eigenvalue = (double *)malloc(num_states*sizeof(double));

  // The rotation matrix (needed for diagonalisation)
  rotation   = (double *)malloc(2*num_states*num_states*sizeof(double));

  /* ----------------------------------------------------
   | Initialise the iterative search for the lowest     |
   | num_states eigenvalues.                            |
   ---------------------------------------------------- */
  printf("Starting iterative search for eigenvalues\n\n");
  printf("+-----------+----------------+-----------------+\n");
  printf("| iteration |     energy     |  energy change  |\n");
  printf("+-----------+----------------+-----------------+\n");

  init_cpu_time = clock();

  // We start from a random guess for trial_wvfn
  // this routine is in libhamiltonian
  c_randomise_state(num_pw,num_states,trial_wvfn);

  // All the wavefunctions should be normalised and orthogonal to each other
  // at every iteration. We enforce this in the initial random state here.
  orthonormalise(num_pw,num_states,trial_wvfn);

  // Apply the H to this state, store the result H.wvfn in gradient. As yet this is
  // unconstrained, i.e. following this gradient will break orthonormality.
  c_apply_H(num_pw,num_states,trial_wvfn,H_kinetic,H_local,gradient);

  // Compute the eigenvalues, i.e. the Rayleigh quotient for each eigenpair
  // Note that we don't compute a denominator here because our trial states
  // are normalised.
  int offset = 0;
  for (nb=0;nb<num_states;nb++) {
    offset = 2*nb*num_pw;
    eigenvalue[nb] = 0.0;
    for (i=0;i<num_pw;i++) {
      // The complex dot-product is conjg(trial_wvfn)*gradient
      // NB the eigenvalue is the real part of the product, so we only compute that
      eigenvalue[nb] += trial_wvfn[offset+2*i]*gradient[offset+2*i];
      eigenvalue[nb] += trial_wvfn[offset+2*i+1]*gradient[offset+2*i+1];
    }
  }

  // Energy is the sum of the eigenvalues.
  energy = 0.0;
  for (nb=0;nb<num_states;nb++) { energy += eigenvalue[nb]; }

  printf("|  Initial  | % #14.8g |                 |\n",energy);
  
  // In case of problems, we cap the total number of iterations
  max_iter = 40000;

  /* ----------------------------------------------------
   | Begin the iterative search for eigenvalues         |
   ---------------------------------------------------- */

  for (iter=1;iter<=max_iter;iter++) {
      
      prev_energy = energy; // book keeping
    
      // The constrained gradient is H.wvfn - (wvfn.H.wvfn)*wvfn
      // -- i.e. it is orthogonal to wvfn which we enforce by 
      // calling the routine below. Remember H.wvfn is already
      // stored as gradient. You need to complete this function
      orthogonalise(num_pw,num_states,gradient,trial_wvfn);

      // The steepest descent search direction is minus the gradient
      for (i=0;i<2*num_pw*num_states;i++) { search_direction[i] = -gradient[i]; }
     
      // Any modifications to the search direction go here, e.g.
      // preconditioning, implementation of conjugate gradients etc.

      // Search along this direction for the best approx. eigenvectors, i.e. the lowest energy.
      line_search(num_pw,num_states,trial_wvfn,H_kinetic,H_local,search_direction,gradient,eigenvalue,&energy);

      // Check convergence
      if(fabs(prev_energy-energy)<energy_tol) {
	  printf("Eigenvalues converged\n");
	  break;
      }

      // Energy is the sum of the eigenvalues
      energy = 0.0;
      for (nb=0;nb<num_states;nb++) { energy += eigenvalue[nb]; }
      printf("|     %4d  | % #14.8g |  % #14.8g |\n",iter,energy,prev_energy-energy); 
      
  }   // end of main iterative loop
  
  curr_cpu_time = clock();

  iter_cpu_time = curr_cpu_time-init_cpu_time;

  printf("Iterative search took %e secs\n\n",(double)(iter_cpu_time)/(double)CLOCKS_PER_SEC);

  /* If you have multiple states, you may get a linear combination of them rather than the
     pure states. This can be fixed by computing the Hamiltonian matrix in the basis of the
     trial states (from trial_wvfn and gradient), and then diagonalising that matrix.

     In other words, we rotate the states to be as near as possible to the true eigenstates 

     This can be done in the "diagonalise" routine, BUT YOU NEED TO COMPLETE IT            */
    
  // diagonalise(num_pw,num_states,trial_wvfn,gradient,eigenvalue,rotation);


  // Finally summarise the results - we renumber the states to start at 1
  printf("=============== FINAL RESULTS ===============\n");
  printf("State                     Eigenvalue         \n");
  printf("                Iterative            Exact   \n");
  for (nb=0;nb<num_states;nb++) {
    printf("   %d   % #18.8g  % #18.8g \n",1+nb,eigenvalue[nb],full_eigenvalue[nb]);
  }

  printf("---------------------------------------------\n");
  printf("Energy % #18.8g  % #18.8g\n\n",energy,exact_energy);
  printf("---------------------------------------------\n");
  printf("Time taken (s) % 10.5g      % 10.5g\n",(double)(iter_cpu_time)/(double)(CLOCKS_PER_SEC),
	 (double)(exact_cpu_time)/(double)(CLOCKS_PER_SEC));
  printf("=============================================\n\n");

  output_results(num_pw, num_states, H_local, trial_wvfn);

  // Release memory
  free(H_kinetic);
  free(H_local);
  free(full_eigenvalue);
  free(trial_wvfn);
  free(gradient);
  free(search_direction);

  exit(EXIT_SUCCESS);

}


// -- YOU WILL NEED TO FINISH THESE FUNCTIONS --

void exact_diagonalisation(int num_pw,double *H_kinetic,double *H_local,double *full_eigenvalue) {
/* |-------------------------------------------------|
   | This subroutine takes a compact representation  |
   | of the matrix H, constructs the full H, and     |
   | diagonalises to get the whole eigenspectrum.    |
   |-------------------------------------------------| */
  double *full_H;
  double *lapack_cmplx_work;
  double *lapack_real_work;
  int lapack_lwork;
  int i;
  int status;
  char jobz;
  char uplo;

  // First we allocate memory for and construct the full Hamiltonian
  full_H = (double *)malloc(2*num_pw*num_pw*sizeof(double));

  // This routine is in libhamiltonian
  c_construct_full_H(num_pw,H_kinetic,H_local,full_H);

  // Use LAPACK to get eigenvalues and eigenvectors
  // NB H is Hermitian (but not packed)
  lapack_real_work = (double *)malloc((3*num_pw-2)*sizeof(double));

  lapack_lwork = 2*num_pw-1;
  lapack_cmplx_work = (double *)malloc(2*lapack_lwork*sizeof(double));
  
  for (i=0;i<3*num_pw-2;i++) { lapack_real_work[i] = 0.0; }
  for (i=0;i<2*lapack_lwork;i++) { lapack_cmplx_work[i] = 0.0; }

  jobz = 'V';
  uplo ='U';
  zheev_(&jobz,&uplo,&num_pw,full_H,&num_pw,full_eigenvalue,lapack_cmplx_work, 
	 &lapack_lwork,lapack_real_work,&status);
  if ( status != 0 ) { 
    printf("Error with zheev in exact_diagonalisation\n");
    exit(EXIT_FAILURE);
  }

  // Deallocate memory
  free(lapack_real_work);
  free(lapack_cmplx_work);
  free(full_H);

  return;

}


void orthogonalise(int num_pw,int num_states, double *state, double *ref_state) {
/* |-------------------------------------------------|
   | This subroutine takes a set of states and       |
   | orthogonalises them to a set of reference       |
   | states.                                         |
   |-------------------------------------------------| */
   char transA;
   char transB;
   double *overlap;
   int offset1;
   int offset2;
   int nb1;
   int nb2;
   int np;
   
   overlap = (double *)malloc(2);
   
  /* |---------------------------------------------------------------------|
     | You need to:                                                        |
     |                                                                     |
     | Compute the overlap of ref_state nb1 with state nb2                 |
     | (use the generalised "dot product" for complex vectors)             |
     |                                                                     |
     | Remove the overlapping parts of ref_state nb1 from state nb2        |
     |---------------------------------------------------------------------| */

   for (nb2=0;nb2<num_states;nb2++) {
     offset2 = 2*nb2*num_pw;
     for (nb1=0;nb1<num_states;nb1++) {
       offset1 = 2*nb1*num_pw;
       overlap[0] = 0.0;
       overlap[1] = 0.0;
       for (np=0;np<num_pw;np++){

	 // The complex dot-product is conjg(ref_state)*state
	 
         // First compute the real part of the overlap
	 overlap[0] += ref_state[offset1+2*np]*state[offset2+2*np];
	 overlap[0] += ref_state[offset1+2*np+1]*state[offset2+2*np+1];

 	 // Now compute the imaginary part of the overlap
 	 overlap[1] += ref_state[offset1+2*np]*state[offset2+2*np+1];
 	 overlap[1] -= ref_state[offset1+2*np+1]*state[offset2+2*np];

       }
       // Remove the overlapping component from state
       for (np=0;np<num_pw;np++){
	 state[offset2+2*np]   -= overlap[0]*ref_state[offset1+2*np]-overlap[1]*ref_state[offset1+2*np+1];
	 state[offset2+2*np+1] -= overlap[0]*ref_state[offset1+2*np+1] + overlap[1]*ref_state[offset1+2*np];
       }
     }
   }
   
  
  return;

}

void precondition(int num_pw,int num_states,double *search_direction,double *trial_wvfn,double *H_kinetic) {
/* |-------------------------------------------------|
   | This subroutine takes a search direction and    |
   | applies a simple kinetic energy-based           |
   | preconditioner to improve the conditioning of   |
   | the eigenvalue search.                          |
   |-------------------------------------------------| */
  int np,nb;
  int offset;
  double kinetic_eigenvalue;

  // Delete these lines once you've coded this subroutine
  printf("Subroutine precondition has not been written yet\n");
  exit(EXIT_FAILURE);
    
  for (nb=0;nb<num_states;nb++) {
    /* |---------------------------------------------------------------------|
       | You need to compute the kinetic energy "eigenvalue" for state nb.   |
       | We don't have the true wavefunction yet, but our best guess is in   |
       | trial_wvfn so we estimate the kinetic energy Ek as:                 |
       |                                                                     |
       |     E_k = trial_wvfn^+ H_kinetic trial_wvfn                         |
       |                                                                     |
       | where "^+" means take the Hermitian conjugate (transpose it and     |
       | take the complex conjugate of each element). H_kinetic is a         |
       | diagonal matrix, so rather than store a num_pw x num_pw matrix with |
       | most elements being zero, we instead just store the num_pw non-zero |
       | elements in a 1D array. Thus the nth element of the result of       |
       | operating with the kinetic energy operator on the wavefunction is:  |
       |                                                                     |
       |     (H_kinetic trial_wvfn)(n) = H_kinetic(n)*trial_wvfn(n)          |
       |                                                                     |
       |---------------------------------------------------------------------| */
    for (np=0;np<num_pw;np++) {
      /* |---------------------------------------------------------------------|
         | You need to compute and apply the preconditioning, using the        |
         | estimate of trial_wvfn's kinetic energy computed above and the      |
         | kinetic energy associated with each plane-wave basis function       |
         |---------------------------------------------------------------------| */
      
    }
  }
  
  return;
 
}


void diagonalise(int num_pw,int num_states, double *state,double *H_state, double *eigenvalues, double *rotation) {
/* |-------------------------------------------------|
   | This subroutine takes a set of states and       |
   | H acting on those states, and transforms the    |
   | states to diagonalise <state|H|state>.          |
   |-------------------------------------------------| */
   int nb1,nb2,i,status;
   int optimal_size;
   int offset1,offset2;

   double *lapack_cmplx_work;
   double *lapack_real_work;
   int lapack_lwork;
   char jobz;
   char uplo;


  // Delete these lines once you've coded this subroutine
  printf("Subroutine diagonalise has not been written yet\n");
  exit(EXIT_FAILURE);

   // Compute the subspace H matrix and store in rotation array
   for (nb2=0;nb2<num_states;nb2++) {
     offset2 = 2*nb2*num_pw;
     for (nb1=0;nb1<num_states;nb1++) {
       offset1 = 2*nb1*num_pw;
       rotation[2*nb2*num_states+2*nb1]   = 0.0;
       rotation[2*nb2*num_states+2*nb1+1] = 0.0;
       for (i=0;i<num_pw;i++) {
         // The complex dot-product a.b is conjg(a)*b
	 
 	 // First compute the real part of the subspace matrix and store in rotation
 	 rotation[2*nb2*num_states+2*nb1] += state[offset1+2*i]*H_state[offset2+2*i];
 	 rotation[2*nb2*num_states+2*nb1] += state[offset1+2*i+1]*H_state[offset2+2*i+1];

 	 // Now compute the imaginary part of the subspace matrix and store in rotation
 	 rotation[2*nb2*num_states+2*nb1+1] += state[offset1+2*i]*H_state[offset2+2*i+1];
 	 rotation[2*nb2*num_states+2*nb1+1] -= state[offset1+2*i+1]*H_state[offset2+2*i];

       }
     }
   }

 
   // Use LAPACK to get eigenvalues and eigenvectors
   // NB H is Hermitian (but not packed)

   // Diagonalise to get eigenvectors and eigenvalues

   // zero the work array

   // Use LAPACK to diagonalise the H in this subspace
   // NB H is Hermitian                                               

   // Deallocate workspace memory

   
   // Finally apply the diagonalising rotation to state
   // (and also to H_state, to keep it consistent with state)
   transform(num_pw,num_states,state,rotation);
   transform(num_pw,num_states,H_state,rotation);

   return;
        
}

/* |-------------------------------------------------------|
   | -- THE FOLLOWING SUBROUTINES ARE ALREADY WRITTEN --   |
   |      (you may wish to optimise them though)           |
   |-------------------------------------------------------| */

void orthonormalise(int num_pw,int num_states,double *state) {
/* |-------------------------------------------------|
   | This subroutine takes a set of states and       |
   | orthonormalises them.                           |
   |-------------------------------------------------| */
   double *overlap;
   int    nb2,nb1,np;
   int    status;
   int    offset1,offset2;

   overlap = (double *)malloc(2*num_states*num_states*sizeof(double));

   // Compute the overlap matrix (using 1D storage)
   for (nb2=0;nb2<num_states;nb2++) {
     offset2 = 2*nb2*num_pw;
     for (nb1=0;nb1<num_states;nb1++) {
       offset1 = 2*nb1*num_pw;
       overlap[2*nb2*num_states+2*nb1]   = 0.0;
       overlap[2*nb2*num_states+2*nb1+1] = 0.0;
       for (np=0;np<num_pw;np++){
	 // The complex dot-product is conjg(trial_wvfn)*gradient
	 
         // First compute the real part of the overlap
	 overlap[2*nb2*num_states+2*nb1] += state[offset1+2*np]*state[offset2+2*np];
	 overlap[2*nb2*num_states+2*nb1] += state[offset1+2*np+1]*state[offset2+2*np+1];

 	 // Now compute the imaginary part of the overlap
 	 overlap[2*nb2*num_states+2*nb1+1] += state[offset1+2*np]*state[offset2+2*np+1];
 	 overlap[2*nb2*num_states+2*nb1+1] -= state[offset1+2*np+1]*state[offset2+2*np];

       }
     }
   }

   // Compute orthogonalising transformation

   // First compute Cholesky (U.U^H) factorisation of the overlap matrix
   char uplo='U';
   zpotrf_(&uplo,&num_states,overlap,&num_states,&status);
   if ( status != 0 ) { 
     printf("zpotrf failed in orthonormalise\n");
     exit(EXIT_FAILURE);
   }

   // invert this upper triangular matrix                                       
   char jobz='N';
   ztrtri_(&uplo,&jobz,&num_states,overlap,&num_states,&status);
   if(status!=0) {
     printf("ztrtri failed in orthonormalise\n");
     exit(EXIT_FAILURE);
   }

   // Set lower triangle to zero - N.B. column-major
   for (nb2=0;nb2<num_states;nb2++) {
     for (nb1=nb2+1;nb1<num_states;nb1++) {
       overlap[2*nb2*num_states+2*nb1]   = 0.0;
       overlap[2*nb2*num_states+2*nb1+1] = 0.0;
     }
   }

   // overlap array now contains the (upper triangular) orthonormalising transformation
   transform(num_pw,num_states,state,overlap);

   free(overlap);

   return;
    
}

void transform(int num_pw,int num_states,double *state,double *transformation) {
/* |-------------------------------------------------|
   | This subroutine takes a set of states and       |
   | applies a linear transformation to them.        |
   |-------------------------------------------------| */
   int nb2,nb1,np,i;
   int offset1,offset2;
   double *new_state;

   new_state=(double *)malloc(2*num_pw*num_states*sizeof(double));

   // Apply transformation to state and H_state
   for (i=0;i<2*num_pw*num_states;i++) { new_state[i] = 0.0; }
 
   for (nb2=0;nb2<num_states;nb2++) {
     offset2 = 2*nb2*num_pw;
     for (nb1=0;nb1<num_states;nb1++) {
       offset1 = 2*nb1*num_pw;
       for (np=0;np<num_pw;np++) {
	 // Here there is no conjugation of the "state" array
	 
	 // Compute real part of new_state
	 new_state[offset1+2*np] += state[offset2+2*np]*transformation[2*nb1*num_states+2*nb2]; 
	 new_state[offset1+2*np] -= state[offset2+2*np+1]*transformation[2*nb1*num_states+2*nb2+1];

	 // Compute imaginary part of new_state
	 new_state[offset1+2*np+1] += state[offset2+2*np]*transformation[2*nb1*num_states+2*nb2+1]; 
	 new_state[offset1+2*np+1] += state[offset2+2*np+1]*transformation[2*nb1*num_states+2*nb2]; 
       }
     }
   }

   for (i=0;i<2*num_pw*num_states;i++) {state[i] = new_state[i];}

   free(new_state);

   return;

}    
		    

void line_search(int num_pw,int num_states,double *approx_state,
		 double *H_kinetic, double *H_local, double *direction,
		 double *gradient,double *eigenvalue,double *energy) {
/*  |-------------------------------------------------|
    | This subroutine takes an approximate eigenstate |
    | and searches along a direction to find an       |
    | improved approximation.                         |
    |-------------------------------------------------| */
    double epsilon;
    double tmp_energy;
    double step;
    double opt_step;
    double *tmp_state;
    double d2E_dstep2;
    double best_step;
    double best_energy;
    double denergy_dstep;
    double mean_norm,inv_mean_norm,tmp_sum;
    int i,loop,nb,np,offset;

    // C doesn't have a nice epsilon() function like Fortran, so
    // we use a lapack routine for this.
    char arg='e';
    double dlamch_(char *arg);
    epsilon = dlamch_(&arg);
    
    // To try to keep a convenient step length, we reduce the size of the search direction
    mean_norm = 0.0;
    for (nb=0;nb<num_states;nb++) {
	offset=2*nb*num_pw;
	tmp_sum = 0.0;
	for (np=0;np<num_pw;np++) {
	    tmp_sum += pow(direction[offset+2*np],2);
	    tmp_sum += pow(direction[offset+2*np+1],2);
	}
	tmp_sum    = sqrt(tmp_sum);
	mean_norm += tmp_sum;
    }
    mean_norm     = mean_norm/(double)num_states;
    inv_mean_norm = 1.0/mean_norm;

    for(i=0;i<2*num_pw*num_states;i++) { direction[i] = direction[i]*inv_mean_norm; }
    
    // The rate-of-change of the energy is just 2*Re{direction.gradient}
    denergy_dstep = 0.0;
    for (nb=0;nb<num_states;nb++) {
	offset = 2*num_pw*nb;
	tmp_sum = 0.0;
	for (np=0;np<num_pw;np++) {
	  // The complex dot-product is conjg(trial_wvfn)*gradient
	  // tmp_sum is the real part, so we only compute that
	  tmp_sum += direction[offset+2*np]*gradient[offset+2*np];
	  tmp_sum += direction[offset+2*np+1]*gradient[offset+2*np+1];
	}
	denergy_dstep += 2.0*tmp_sum;
    }
    
    tmp_state = (double *)malloc(2*num_pw*num_states*sizeof(double));
    
    best_step   = 0.0;
    best_energy = *energy;

    // First take a trial step in the direction
    step = trial_step;
    
    // We find a trial step that lowers the energy:
    for (loop=0;loop<10;loop++) {
	
	for (i=0;i<2*num_pw*num_states;i++) {
	    tmp_state[i]   = approx_state[i] + step*direction[i];
	}
	
	orthonormalise(num_pw,num_states,tmp_state);
	
	// Apply the H to this state
	c_apply_H(num_pw,num_states,tmp_state,H_kinetic,H_local,gradient);
	
	// Compute the new energy estimate
	tmp_energy = 0.0;
	for (nb=0;nb<num_states;nb++) {
	    offset = 2*num_pw*nb;
	    tmp_sum = 0.0;
	    for (np=0;np<num_pw;np++) {
	      // The complex dot-product a.b is conjg(a)*b
	      // tmp_sum is the real part, so we just compute that
	      tmp_sum +=tmp_state[offset+2*np]*gradient[offset+2*np];
	      tmp_sum +=tmp_state[offset+2*np+1]*gradient[offset+2*np+1];
	    }
	    tmp_energy += tmp_sum;
	}
	
	if (tmp_energy<*energy) {
	    break;
	}else {
	    d2E_dstep2 = (tmp_energy - *energy - step*denergy_dstep )/(step*step);
	    if(d2E_dstep2<0.0) {
		if(tmp_energy<*energy) {
		    break;
		}  else {
		    step = step/4.0;
		}
	    } else {
		step  = -denergy_dstep/(2*d2E_dstep2);
	    }
	    
	}
    }
    
    if (tmp_energy<best_energy) {
	best_step   = step;
	best_energy = tmp_energy;
    }
    
    // We now have the initial eigenvalue, the initial gradient, and a trial step
    // -- we fit a parabola, and jump to the estimated minimum position
    // Set default step and energy
    d2E_dstep2 = (tmp_energy - *energy - step*denergy_dstep )/(step*step);
    
    if(d2E_dstep2<0.0) {
	// Parabolic fit gives a maximum, so no good
	printf("** Warning, parabolic stationary point is a maximum **\n");
	
	if(tmp_energy<*energy) {
	    opt_step = step;
	}else {
	    opt_step = 0.1*step;
	}
    } else {
	opt_step  = -denergy_dstep/(2.0*d2E_dstep2);
    }
    
    
    //    e = e0 + de*x + c*x**2
    // => c = (e - e0 - de*x)/x**2
    // => min. at -de/(2c)
    //
    //    de/dx = de + 2*c*x
    
    
    for (i=0;i<2*num_pw*num_states;i++) { approx_state[i] += opt_step*direction[i];}
    
    orthonormalise(num_pw,num_states,approx_state);
    
    // Apply the H to this state
    c_apply_H(num_pw,num_states,approx_state,H_kinetic,H_local,gradient);
    
    // Compute the new energy estimate
    *energy = 0.0;
    for (nb=0;nb<num_states;nb++) {
	offset = 2*num_pw*nb;
	tmp_sum = 0.0;
	for (np=0;np<num_pw;np++) {
	  // The complex dot-product a.b is conjg(a)*b
	  // tmp_sum is just the real part, so we only compute that
	  tmp_sum += approx_state[offset+2*np]*gradient[offset+2*np];
	  tmp_sum += approx_state[offset+2*np+1]*gradient[offset+2*np+1];
	}
	eigenvalue[nb] = tmp_sum;
	*energy += tmp_sum;
    }
    
    // This ought to be the best, but check...
    if (*energy>best_energy) {
	// if(best_step>0.0_dp) then
	if (fabs(best_step-epsilon)>0.0) { // roughly machine epsilon in double precision
	    
	    for (i=0;i<2*num_pw*num_states;i++) {
		approx_state[i] += best_step*direction[i];
	    }
	    
	    orthonormalise(num_pw,num_states,approx_state);
	    
	    // Apply the H to this state
	    c_apply_H(num_pw,num_states,approx_state,H_kinetic,H_local,gradient);
	    
	    // Compute the new energy estimate
	    *energy = 0.0;
	    for (nb=0;nb<num_states;nb++) {
		offset = 2*num_pw*nb;
		tmp_sum = 0.0;
		for (np=0;np<num_pw;np++) {
		  // The complex dot-product a.b is conjg(a)*b
		  // tmp_sum is just the real part, so we only compute that
		  tmp_sum += approx_state[offset+2*np]*gradient[offset+2*np];
		  tmp_sum += approx_state[offset+2*np+1]*gradient[offset+2*np+1];
		}
		eigenvalue[nb] = tmp_sum;
		*energy += tmp_sum;
	    }
	} else {
	    printf("Oh dear: %f\n",best_step);
	    printf("Problem with line search\n");
	    exit(EXIT_FAILURE);
	}
    }
    
    
    //       printf(" %f %d %f <- test2\n",opt_step,step,energy);
    
    // We'll use this step as the basis of our trial step next time
    // trial_step = 2*opt_step;
    
    free(tmp_state);
    return;
    
}

void output_results(int num_pw, int num_states, double* H_local,
                    double* wvfn) {

  FILE* potential = fopen("pot.dat", "w");

  for (int i = 0; i < num_pw; ++i) {
    fprintf(potential, "%.12g\t%.12g\n", i / (double)(num_pw), H_local[i]);
  }

  fclose(potential);

  fftw_complex* realspace_wvfn =
      (fftw_complex*)fftw_malloc(num_pw * sizeof(fftw_complex));
  fftw_complex* tmp_wvfn = (fftw_complex*)fftw_malloc(num_pw * sizeof(fftw_complex));

  fftw_plan plan =
      fftw_plan_dft_1d(num_pw, tmp_wvfn, realspace_wvfn, FFTW_FORWARD, FFTW_ESTIMATE);

  for (int nb = 0; nb < num_states; ++nb) {
    char filename[15];
    snprintf(filename, 15, "wvfn_%.4i.dat", nb);
    FILE* wvfn_file = fopen(filename, "w");

    int offset = 2*nb * num_pw;
    memcpy(tmp_wvfn, &wvfn[offset], 2*num_pw*sizeof(double));

    fftw_execute_dft(plan, tmp_wvfn, realspace_wvfn);

    for (int np = 0; np < num_pw; ++np) {
      fprintf(wvfn_file, "%.12g\t%.12g\n", np / (double)(num_pw),
              realspace_wvfn[np][0]);
    }

    fclose(wvfn_file);
  }
  fftw_destroy_plan(plan);
  fftw_free(realspace_wvfn);
  fftw_free(tmp_wvfn);
}
