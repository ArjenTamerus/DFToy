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
#include <complex.h>
#include <mpi.h>
#include <fftw3.h>
#include "interface.h"
#include "trace.h"

// MPI variables
int world_size, world_rank;

// Main program unit
int main(int argc, char **argv)
{

	/* ----------------------------------------------------
		 | Declare variables                                |
		 ---------------------------------------------------- */

	//double *H_kinetic;        // the kinetic energy operator
	//double *H_local;          // the local potential operator
	int     num_wavevectors;  // no. wavevectors in the basis
	int     num_pw;           // no. plane waves
	int     num_states;       // no. eigenstates req'd

	fftw_complex *trial_wvfn;       // current best guess at wvfn
	fftw_complex *gradient;         // gradient of energy
	fftw_complex *search_direction; // direction to search along
	fftw_complex *prev_search_direction; // last direction searched along (for CG)

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
	fftw_complex *rotation;

	// Timing variables
	clock_t init_cpu_time,curr_cpu_time,exact_cpu_time,iter_cpu_time;

	/* ---------------------
		 | Initialise system |
		 --------------------*/
	// MPI standard stuff
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

	// No. nonzero wavevectors "G" in our wavefunction expansion
	num_wavevectors = 100;

	// No. eigenstates to compute
	num_states = 1;

	// Update num_wavevectors and num_states from cmd line
	// Don't care about error handling
	// Just read `./eigensolver [num_wavevectors [num_states]]
	// TODO MPI-ify
	//mpi_printf(world_rank,"CMD: %s", argv[0]);
	if (argc > 1) {
		//mpi_printf(world_rank," %s", argv[1]);
		num_wavevectors = (int) strtol(argv[1], NULL, 10);
	}
	if (argc > 2) {
		//mpi_printf(world_rank," %s", argv[2]);
		num_states = (int) strtol(argv[2], NULL, 10);
	}
	//mpi_printf(world_rank,"\n");

	mpi_printf(world_rank,"ntasks: %d\n", world_size);
	// No. plane-waves in our wavefunction expansion. One plane-wave has
	// wavevector 0, and for all the others there are plane-waves at +/- G
	num_pw = num_wavevectors;
	mpi_printf(world_rank,"num_pw: %d\n", num_pw);

	// Catch any nonsensical combinations of parameters
	if (num_states>=num_pw) {
		mpi_printf(world_rank,"Error, num_states must be less than num_pw\n");
		MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
		//exit(EXIT_FAILURE);
	} 

	// Set tolerance on the eigenvalue sum when using an iterative search. 
	// The iterative search will stop when the change in the eigenvalue sum
	// per iteration is less than this tolerance.
	energy_tol = 1.0e-10;

	// Initialise random number generator
	c_init_random_seed();

	//mpi_printf(world_rank,"Initialising Hamiltonian...\n");

	// Initialise and build the Hamiltonian, comprising two terms: the kinetic energy,
	// which is a diagonal matrix in the plane-wave basis (Fourier space); and the local
	// potential energy, which is a diagonal matrix in real space (direct space).
	//
	// The Hamiltonian is built inside libhamiltonian by populating these two arrays.
	// NB these arrays are *real* (not complex)
	H_kinetic  = (double *)TRACEMALLOC(num_pw*sizeof(double));  
	H_local = (double *)TRACEMALLOC(num_pw*sizeof(double));
	c_init_H(num_pw,H_kinetic,H_local);

	/* ----------------------------------------------------
		 | Perform full diagonalisation using LAPACK.       |
		 | You will need to complete the function called    |
		 ---------------------------------------------------- */
	RESET_MEMSTATS();
	//mpi_printf(world_rank,"Starting full diagonalisation...\n\n");

	init_cpu_time    = clock();
	full_eigenvalue = (double *)TRACEMALLOC(num_pw*sizeof(double));
	int diag_mode = get_diag_mode();
	exact_diagonalisation(num_pw, num_states, H_kinetic,H_local,full_eigenvalue, diag_mode);
	curr_cpu_time    = clock();

	exact_cpu_time = curr_cpu_time-init_cpu_time;

	//mpi_printf(world_rank," State         Eigenvalue\n");
	//for (nb=0;nb<num_states;nb++) {
	//	mpi_printf(world_rank,"     %d % #19.10g\n",1+nb,full_eigenvalue[nb]);
	//}

	// Energy is the sum of the eigenvalues of the occupied states
	exact_energy = 0.0;
	for (nb=0;nb<num_states;nb++) { exact_energy += full_eigenvalue[nb]; }
	mpi_printf(world_rank,"Ground state energy: % #16.10g\n",exact_energy);

	mpi_printf(world_rank,"Full diagonalisation: %f\n",(double)exact_cpu_time/(double)CLOCKS_PER_SEC);

	report_mem_stats();
	RESET_MEMSTATS();
	MPI_Finalize();
	return 0;

	// Allocate memory for iterative eigenvector search. Each of the following
	// are stored in column-major (Fortran) order. Each column contains the 
	// plane wave co-efficients for a single particle wavefunction and there
	// are num_states particles.
	//
	trial_wvfn            = (fftw_complex *)TRACEMALLOC(num_pw*num_states*sizeof(fftw_complex));
	gradient              = (fftw_complex *)TRACEMALLOC(num_pw*num_states*sizeof(fftw_complex));
	search_direction      = (fftw_complex *)TRACEMALLOC(num_pw*num_states*sizeof(fftw_complex));
	prev_search_direction = (fftw_complex *)TRACEMALLOC(num_pw*num_states*sizeof(fftw_complex));

	// We have num_states eigenvalue estimates (real) and products (complex)
	eigenvalue = (double *)TRACEMALLOC(num_states*sizeof(double));

	// The rotation matrix (needed for diagonalisation)
	rotation   = (fftw_complex *)TRACEMALLOC(num_states*num_states*sizeof(double));

	/* ----------------------------------------------------
		 | Initialise the iterative search for the lowest     |
		 | num_states eigenvalues.                            |
		 ---------------------------------------------------- */
	mpi_printf(world_rank,"Starting iterative search for eigenvalues\n\n");
	mpi_printf(world_rank,"+-----------+----------------+-----------------+\n");
	mpi_printf(world_rank,"| iteration |     energy     |  energy change  |\n");
	mpi_printf(world_rank,"+-----------+----------------+-----------------+\n");

	init_cpu_time = clock();

	// We start from a random guess for trial_wvfn
	// this routine is in libhamiltonian
	c_randomise_state(num_pw, num_states, trial_wvfn);

	// All the wavefunctions should be normalised and orthogonal to each other
	// at every iteration. We enforce this in the initial random state here.
	orthonormalise(num_pw, num_states, trial_wvfn);

	// Apply the H to this state, store the result H.wvfn in gradient. As yet this is
	// unconstrained, i.e. following this gradient will break orthonormality.
	c_apply_H(num_pw, num_states, trial_wvfn, H_kinetic, H_local, gradient);

	// Compute the eigenvalues, i.e. the Rayleigh quotient for each eigenpair
	// Note that we don't compute a denominator here because our trial states
	// are normalised.
	int offset = 0;
	for (nb=0;nb<num_states;nb++) {
		offset = nb*num_pw;
		eigenvalue[nb] = 0.0;
		for (i=0;i<num_pw;i++) {
			// The complex dot-product is conjg(trial_wvfn)*gradient
			// NB the eigenvalue is the real part of the product, so we only compute that
			//eigenvalue[nb] += creal(trial_wvfn[offset+2*i]*gradient[offset+2*i];
			//eigenvalue[nb] += trial_wvfn[offset+2*i+1]*gradient[offset+2*i+1];
			eigenvalue[nb] += creal(conj(trial_wvfn[offset+i])*gradient[offset+i]);
		}
	}

	// Energy is the sum of the eigenvalues.
	energy = 0.0;
	for (nb=0;nb<num_states;nb++) { energy += eigenvalue[nb]; }

	mpi_printf(world_rank,"|  Initial  | % #14.8g |                 |\n",energy);

	// In case of problems, we cap the total number of iterations
	max_iter = 40000;

	/* ----------------------------------------------------
		 | Begin the iterative search for eigenvalues         |
		 ---------------------------------------------------- */
	int CG_RESET=5;
	double gamma, gTxg=0.0, gTxg_prev=1.0;

	for (iter=1;iter<=max_iter;iter++) {

		prev_energy = energy; // book keeping

		// The constrained gradient is H.wvfn - (wvfn.H.wvfn)*wvfn
		// -- i.e. it is orthogonal to wvfn which we enforce by 
		// calling the routine below. Remember H.wvfn is already
		// stored as gradient. You need to complete this function
		orthogonalise(num_pw,num_states,gradient,trial_wvfn);

		// The steepest descent search direction is minus the gradient
		for (i=0;i<num_pw*num_states;i++) {
			// cannot copy into prev_search_direction *here* because search_direction
			// gets mangled inside the line_search function
			search_direction[i] = -gradient[i];
		}

		// Any modifications to the search direction go here, e.g.
		// preconditioning, implementation of conjugate gradients etc.

		//* PRECON */
		precondition(num_pw, num_states, search_direction, trial_wvfn, H_kinetic);
		orthogonalise(num_pw, num_states, search_direction, trial_wvfn);

		//Always calculate gT*g even on reset iteration - will need as gTxg_prev in
		//next iteration
		//gTxg_prev = gTxg;
		gTxg = 0.0;

		for(nb = 0; nb < num_states; nb++) {
			offset = nb*num_pw;
			for(i=0; i < num_pw; i++) {
				gTxg += creal(conj(search_direction[offset+i])*gradient[offset+i]);
			}
		}

		if (reset_sd!=0) {
			gamma = gTxg/gTxg_prev;
			for (nb=0;nb<num_states;nb++) {
				offset = nb*num_pw;
				for (i=0;i<num_pw;i++) {
					search_direction[offset+i] += gamma*prev_search_direction[offset+i];
				}
			}

			orthogonalise(num_pw,num_states,search_direction,trial_wvfn);
		}

		gTxg_prev = gTxg;
		for(i=0;i<num_pw*num_states;i++) { prev_search_direction[i] = search_direction[i]; }
		// Search along this direction for the best approx. eigenvectors, i.e. the lowest energy.
		line_search(num_pw,num_states,trial_wvfn,H_kinetic,H_local,search_direction,gradient,eigenvalue,&energy);

		// Check convergence
		if(fabs(prev_energy-energy)<energy_tol) {
			mpi_printf(world_rank,"+-----------+----------------+-----------------+\n");
			mpi_printf(world_rank,"Eigenvalues converged\n");
			break;
		}
		if(fabs(prev_energy-energy)<energy_tol) {
			if(reset_sd==0){
				mpi_printf(world_rank,"+-----------+----------------+-----------------+\n");
				mpi_printf(world_rank,"Eigenvalues converged\n");
				break;
			} else {
				reset_sd = 0;
			}
		} else {
			reset_sd = 1;
		}
		// Reset the CG every 5 steps, to prevent it stagnating
		//if (iter%CG_RESET==0) reset_sd = 1;

		// Energy is the sum of the eigenvalues
		energy = 0.0;
		for (nb=0;nb<num_states;nb++) { energy += eigenvalue[nb]; }
		mpi_printf(world_rank,"|     %4d  | % #14.8g |  % #14.8g |\n",iter,energy,prev_energy-energy); 

	}   // end of main iterative loop

	curr_cpu_time = clock();

	iter_cpu_time = curr_cpu_time-init_cpu_time;

	mpi_printf(world_rank,"Iterative search took %e secs\n\n",(double)(iter_cpu_time)/(double)CLOCKS_PER_SEC);

	/* If you have multiple states, you may get a linear combination of them rather than the
		 pure states. This can be fixed by computing the Hamiltonian matrix in the basis of the
		 trial states (from trial_wvfn and gradient), and then diagonalising that matrix.

		 In other words, we rotate the states to be as near as possible to the true eigenstates 

		 This can be done in the "diagonalise" routine, BUT YOU NEED TO COMPLETE IT            */

	 diagonalise(num_pw,num_states,trial_wvfn,gradient,eigenvalue,rotation);

	report_mem_stats();


	// Finally summarise the results - we renumber the states to start at 1
	mpi_printf(world_rank,"=============== FINAL RESULTS ===============\n");
	mpi_printf(world_rank,"State                     Eigenvalue         \n");
	mpi_printf(world_rank,"                Iterative            Exact   \n");
	for (nb=0;nb<num_states;nb++) {
		mpi_printf(world_rank,"   %d   % #18.8g  % #18.8g \n",1+nb,eigenvalue[nb],full_eigenvalue[nb]);
	}

	mpi_printf(world_rank,"---------------------------------------------\n");
	mpi_printf(world_rank,"Energy % #18.8g  % #18.8g\n\n",energy,exact_energy);
	mpi_printf(world_rank,"---------------------------------------------\n");
	mpi_printf(world_rank,"Time taken (s) % 10.5g      % 10.5g\n",(double)(iter_cpu_time)/(double)(CLOCKS_PER_SEC),
			(double)(exact_cpu_time)/(double)(CLOCKS_PER_SEC));
	mpi_printf(world_rank,"=============================================\n\n");

	output_results(num_pw, num_states, H_local, trial_wvfn);

	// Release memory
	TRACEFREE(H_kinetic);
	TRACEFREE(H_local);
	TRACEFREE(full_eigenvalue);
	TRACEFREE(trial_wvfn);
	TRACEFREE(gradient);
	TRACEFREE(search_direction);

	// Your standard MPI goodbye
	MPI_Finalize();

	exit(EXIT_SUCCESS);

}


// -- YOU WILL NEED TO FINISH THESE FUNCTIONS --
//
//void my_construct_full_H(int num_pw, double *H_kinetic, double *H_local, fftw_complex *full_H)
//{
//	fftw_complex *tmp_state1, *tmp_state2;
//	fftw_plan plan_forward, plan_backward;
//	int np1, np2;
//
//	//memset(full_H, num_pw*num_pw*sizeof(fftw_complex), 0);	
//	for (np1 = 0; np1 < num_pw*num_pw; np1++) {
//		full_H[np1] = (0+0*I);
//	}
//
//	tmp_state1 = TRACEFFTW_MALLOC(num_pw*sizeof(fftw_complex));
//	tmp_state2 = TRACEFFTW_MALLOC(num_pw*sizeof(fftw_complex));
//	//memset(tmp_state2, num_pw*sizeof(fftw_complex), 0);	
//	for (np1 = 0; np1 < num_pw; np1++) {
//		tmp_state2[np1] = (0+0*I);
//	}
//
//	plan_forward = fftw_plan_dft_1d(num_pw, tmp_state1, tmp_state2, FFTW_FORWARD, FFTW_ESTIMATE);
//	plan_backward = fftw_plan_dft_1d(num_pw, tmp_state1, tmp_state2, FFTW_BACKWARD, FFTW_ESTIMATE);
//
//	for(np1 = 0; np1 < num_pw; np1++) {
//		//memset(tmp_state1, num_pw*sizeof(fftw_complex), 0);	
//		for (np2 = 0; np2 < num_pw; np2++) {
//			tmp_state1[np2] = (0+0*I);
//		}
//		tmp_state1[np1] = (1+0*I);
//
//		fftw_execute_dft(plan_forward, tmp_state1, tmp_state2);
//
//		for(np2 = 0; np2 < num_pw; np2++)
//		{
//			tmp_state2[np2] = H_local[np2]*tmp_state2[np2]/(((double)1.0)*num_pw);
//		}
//
//		fftw_execute_dft(plan_backward, tmp_state2, tmp_state1);
//
//		for(np2 = 0; np2 < num_pw; np2++)
//		{
//			full_H[np1*num_pw + np2] = tmp_state1[np2];
//		}
//	}
//
//	fftw_destroy_plan(plan_forward);
//	fftw_destroy_plan(plan_backward);
//
//	TRACEFFTW_FREE(tmp_state1);
//	TRACEFFTW_FREE(tmp_state2);
//
//	// Add contribution from kinetic term
//	for(int i=1;i<num_pw;i++) {
//		full_H[i*num_pw+i] += (H_kinetic[i]+0i);
//	}
//}

void exact_diagonalisation(int num_pw, int num_states, double *H_kinetic, double *H_local, double *full_eigenvalue, int diag_mode) {
	/* |-------------------------------------------------|
		 | This subroutine takes a compact representation  |
		 | of the matrix H, constructs the full H, and     |
		 | diagonalises to get the whole eigenspectrum.    |
		 |-------------------------------------------------| */

	switch (diag_mode) {
		case 0:
			diag_zheev(num_pw, H_kinetic, H_local, full_eigenvalue);
			break;
		case 1:
			diag_zheevd(num_pw, H_kinetic, H_local, full_eigenvalue);
			break;
		case 2:
			diag_zheevr(num_pw, num_states, H_kinetic, H_local, full_eigenvalue);
			break;
		case 3:
			diag_pzheev(num_pw, H_kinetic, H_local, full_eigenvalue);
			break;
		case 4:
			diag_pzheevd(num_pw, H_kinetic, H_local, full_eigenvalue);
			break;
		case 5:
			diag_pzheevr(num_pw, num_states, H_kinetic, H_local, full_eigenvalue);
			break;
		case 8:
			diag_elpa(num_pw, H_kinetic, H_local, full_eigenvalue);
			break;
		default:
			diag_zheev(num_pw, H_kinetic, H_local, full_eigenvalue);
			break;
	};

}

void orthogonalise(int num_pw,int num_states, fftw_complex *state, fftw_complex *ref_state) {
	/* |-------------------------------------------------|
		 | This subroutine takes a set of states and       |
		 | orthogonalises them to a set of reference       |
		 | states.                                         |
		 |-------------------------------------------------| */
	fftw_complex overlap;
	int ref_state_offset;
	int state_offset;
	int nb1;
	int nb2;
	int np;

	int local_ref_state_offset;
	int local_state_offset;

	/* |---------------------------------------------------------------------|
		 | You need to:                                                        |
		 |                                                                     |
		 | Compute the overlap of ref_state nb1 with state nb2                 |
		 | (use the generalised "dot product" for complex vectors)             |
		 |                                                                     |
		 | Remove the overlapping parts of ref_state nb1 from state nb2        |
		 |---------------------------------------------------------------------| */

	for (nb2=0;nb2<num_states;nb2++) {
		int state_offset = nb2*num_pw;
		for (nb1=0;nb1<num_states;nb1++) {
			int ref_state_offset = nb1*num_pw;

			overlap = (0+0*I);

			// Calculate overlap
			// Dot. Prod. = SUM_i(cplx_conj(a)_i*b_i)
			for (np=0; np < num_pw; np++) {
				local_ref_state_offset = ref_state_offset+np;
				local_state_offset = state_offset+np;

				overlap += conj(ref_state[local_ref_state_offset])*state[local_state_offset];

			}
			
			// remove overlap from state
			for (np=0; np < num_pw; np++) {
				local_ref_state_offset = ref_state_offset+np;
				local_state_offset = state_offset+np;

				state[local_state_offset] -= overlap*ref_state[local_ref_state_offset];
			}
		}
	}

}

void precondition(int num_pw, int num_states, fftw_complex *search_direction, fftw_complex *trial_wvfn, double *H_kinetic) {
	/* |-------------------------------------------------|
		 | This subroutine takes a search direction and    |
		 | applies a simple kinetic energy-based           |
		 | preconditioner to improve the conditioning of   |
		 | the eigenvalue search.                          |
		 |-------------------------------------------------| */
	int np,nb;
	int offset;
	double kinetic_eigenvalue;
	double x, tmp; 

	//fftw_double *trial_wvfn = (fftw_complex *)trial_wvfn_d;

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

		offset = nb*num_pw;
		kinetic_eigenvalue = 0.0;

		for (np=0;np<num_pw;np++) {
			//kinetic_eigenvalue += H_kinetic[np] * trial_wvfn[offset+np] * trial_wvfn[offset+np];
			kinetic_eigenvalue += H_kinetic[np] * creal(trial_wvfn[offset+np]) * creal(trial_wvfn[offset+np])+ H_kinetic[np] *cimag(trial_wvfn[offset+np]) * cimag(trial_wvfn[offset+np]);
			//kinetic_eigenvalue += H_kinetic[np] * trial_wvfn[offset+np+1] * trial_wvfn[offset+2*np+1];
		}
		
		for (np=0;np<num_pw;np++) {
			/* |---------------------------------------------------------------------|
				 | You need to compute and apply the preconditioning, using the        |
				 | estimate of trial_wvfn's kinetic energy computed above and the      |
				 | kinetic energy associated with each plane-wave basis function       |
				 |---------------------------------------------------------------------| */

			// x = H_kin/E_kin
			x = H_kinetic[np] / kinetic_eigenvalue;

			// apply f(x) - keeping in mind search_direction is complex
			// f(x) = (8+4x+2x^2+x^3)/(8+4x+2x^2+x^3+x^4)
			// = (8+4x+2x^2+x^3)/(8+4x+2x^2+x^3)*x^4)
			//
			// 8+4x+2x^2+x^3 = 8 + x * (4+ x * (2 + x ))
			tmp = 8.0 + x * (4.0 + x * (2.0 + x));

			// real
			search_direction[offset+np] *= tmp / (tmp + x*x*x*x);
			// complex
			//search_direction[offset+2*np+1] *= tmp / (tmp + x*x*x*x);


		}
	}

}


void diagonalise(int num_pw,int num_states, fftw_complex *state, fftw_complex *H_state, double *eigenvalues, fftw_complex *rotation) {
	/* |-------------------------------------------------|
		 | This subroutine takes a set of states and       |
		 | H acting on those states, and transforms the    |
		 | states to diagonalise <state|H|state>.          |
		 |-------------------------------------------------| */
	int nb1,nb2,i,status;
	int optimal_size;
	int offset1,offset2;

	fftw_complex *lapack_cmplx_work;
	double *lapack_real_work;
	int lapack_lwork;
	char jobz;
	char uplo;

	// Compute the subspace H matrix and store in rotation array
	for (nb2=0;nb2<num_states;nb2++) {
		offset2 = nb2*num_pw;
		for (nb1=0;nb1<num_states;nb1++) {
			offset1 = nb1*num_pw;
			rotation[nb2*num_states+nb1] = (0.0+0.0*I);
			for (i=0;i<num_pw;i++) {

				// The complex dot-product a.b is conjg(a)*b
				rotation[nb2*num_states+nb1] += conj(state[offset1+i])*H_state[offset2+i];

			}
		}
	}


	// Use LAPACK to get eigenvalues and eigenvectors
	// NB H is Hermitian (but not packed)

	// Diagonalise to get eigenvectors and eigenvalues

	lapack_lwork = 2*num_states-1;
	lapack_real_work = TRACECALLOC((3*num_states-2),sizeof(double));
	lapack_cmplx_work = TRACEFFTW_MALLOC(lapack_lwork*sizeof(fftw_complex));

	jobz = 'V';
	uplo = 'U';
	zheev_(&jobz, &uplo, &num_states, rotation, &num_states, eigenvalues, lapack_cmplx_work, &lapack_lwork, lapack_real_work, &status);
	// Use LAPACK to diagonalise the H in this subspace
	// NB H is Hermitian                                               

	// Deallocate workspace memory
	TRACEFREE(lapack_real_work);
	TRACEFFTW_FREE(lapack_cmplx_work);


	// Finally apply the diagonalising rotation to state
	// (and also to H_state, to keep it consistent with state)
	transform(num_pw,num_states,state,rotation);
	transform(num_pw,num_states,H_state,rotation);

}

/* |-------------------------------------------------------|
	 | -- THE FOLLOWING SUBROUTINES ARE ALREADY WRITTEN --   |
	 |      (you may wish to optimise them though)           |
	 |-------------------------------------------------------| */

void orthonormalise(int num_pw, int num_states, fftw_complex *state) {
	/* |-------------------------------------------------|
		 | This subroutine takes a set of states and       |
		 | orthonormalises them.                           |
		 |-------------------------------------------------| */
	fftw_complex *overlap;
	int    nb2,nb1,np;
	int    status;
	int    offset1,offset2;

	overlap = (fftw_complex *)TRACEFFTW_MALLOC(num_states*num_states*sizeof(fftw_complex));

	// Compute the overlap matrix (using 1D storage)
	for (nb2=0;nb2<num_states;nb2++) {
		offset2 = nb2*num_pw;
		for (nb1=0;nb1<num_states;nb1++) {
			offset1 = nb1*num_pw;

			overlap[nb2*num_states+nb1] = (0.0+0.0*I);
			for (np=0;np<num_pw;np++){
				// The complex dot-product is conjg(trial_wvfn)*gradient
				overlap[nb2*num_states+nb1] += conj(state[offset1+np])*state[offset2+np];
			}
		}
	}

	// Compute orthogonalising transformation

	// First compute Cholesky (U.U^H) factorisation of the overlap matrix
	char uplo='U';
	zpotrf_(&uplo,&num_states,overlap,&num_states,&status);
	if ( status != 0 ) { 
		mpi_printf(world_rank,"zpotrf failed in orthonormalise\n");
		exit(EXIT_FAILURE);
	}

	// invert this upper triangular matrix                                       
	char jobz='N';
	ztrtri_(&uplo,&jobz,&num_states,overlap,&num_states,&status);
	if(status!=0) {
		mpi_printf(world_rank,"ztrtri failed in orthonormalise\n");
		exit(EXIT_FAILURE);
	}

	// Set lower triangle to zero - N.B. column-major
	for (nb2=0;nb2<num_states;nb2++) {
		for (nb1=nb2+1;nb1<num_states;nb1++) {
			overlap[nb2*num_states+nb1] = (0.0+0.0*I);
		}
	}

	// overlap array now contains the (upper triangular) orthonormalising transformation
	transform(num_pw, num_states, state, overlap);

	TRACEFFTW_FREE(overlap);

}

void transform(int num_pw, int num_states, fftw_complex *state, fftw_complex *transformation) {
	/* |-------------------------------------------------|
		 | This subroutine takes a set of states and       |
		 | applies a linear transformation to them.        |
		 |-------------------------------------------------| */
	int nb2,nb1,np,i;
	int offset1,offset2;
	fftw_complex *new_state;

	new_state = (fftw_complex *)TRACEFFTW_MALLOC(num_pw*num_states*sizeof(fftw_complex));

	// Apply transformation to state and H_state
	for (i=0;i<num_pw*num_states;i++) { new_state[i] = (0.0+0.0*I); }

	for (nb2=0;nb2<num_states;nb2++) {
		offset2 = nb2*num_pw;
		for (nb1=0;nb1<num_states;nb1++) {
			offset1 = nb1*num_pw;
			for (np=0;np<num_pw;np++) {
				// Here there is no conjugation of the "state" array
				new_state[offset1+np] += state[offset2+np]*transformation[nb1*num_states+nb2];
			}
		}
	}

	for (i=0;i<num_pw*num_states;i++) {state[i] = new_state[i];}

	TRACEFFTW_FREE(new_state);

}


void line_search(int num_pw ,int num_states, fftw_complex *approx_state,
		double *H_kinetic, double *H_local, fftw_complex *direction,
		fftw_complex *gradient, double *eigenvalue, double *energy) {
	/*  |-------------------------------------------------|
			| This subroutine takes an approximate eigenstate |
			| and searches along a direction to find an       |
			| improved approximation.                         |
			|-------------------------------------------------| */
	double epsilon;
	double tmp_energy;
	double step;
	double opt_step;
	fftw_complex *tmp_state;
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

	double *d_direction = (double*) direction;
	// To try to keep a convenient step length, we reduce the size of the search direction
	mean_norm = 0.0;
	for (nb=0;nb<num_states;nb++) {
		offset=nb*num_pw;
		tmp_sum = 0.0;
		for (np=0;np<num_pw;np++) {
			// NOTE apparently taking mean_norm as sum(abs(direction)) converged
			// faster than the original? Why?
			//mean_norm += cabs(direction[offset+np]);
			tmp_sum += pow(cabs(direction[offset+np]),2);
		}
		tmp_sum    = sqrt(tmp_sum);
		mean_norm += tmp_sum;
	}
	mean_norm     = mean_norm/(double)num_states;
	inv_mean_norm = 1.0/mean_norm;

	for(i=0;i<num_pw*num_states;i++) { direction[i] = direction[i]*inv_mean_norm; }

	// The rate-of-change of the energy is just 2*Re{direction.gradient}
	denergy_dstep = 0.0;
	for (nb=0;nb<num_states;nb++) {
		offset = num_pw*nb;
		tmp_sum = 0.0;
		for (np=0;np<num_pw;np++) {
			// The complex dot-product is conjg(trial_wvfn)*gradient
			// tmp_sum is the real part, so we only compute that
			tmp_sum += creal(conj(direction[offset+np])*gradient[offset+np]);
		}
		denergy_dstep += 2.0*tmp_sum;
	}

	tmp_state = (fftw_complex *)TRACEMALLOC(num_pw*num_states*sizeof(fftw_complex));

	best_step   = 0.0;
	best_energy = *energy;

	// First take a trial step in the direction
	step = trial_step;

	// We find a trial step that lowers the energy:
	for (loop=0;loop<10;loop++) {

		for (i=0;i<num_pw*num_states;i++) {
			tmp_state[i]   = approx_state[i] + step*direction[i];
		}

		orthonormalise(num_pw,num_states,tmp_state);

		// Apply the H to this state
		c_apply_H(num_pw,num_states,tmp_state,H_kinetic,H_local,gradient);

		// Compute the new energy estimate
		tmp_energy = 0.0;
		for (nb=0;nb<num_states;nb++) {
			offset = num_pw*nb;
			tmp_sum = 0.0;
			for (np=0;np<num_pw;np++) {
				// The complex dot-product a.b is conjg(a)*b
				// tmp_sum is the real part, so we just compute that
				tmp_sum += creal(conj(tmp_state[offset+np])*gradient[offset+np]);
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
		mpi_printf(world_rank,"** Warning, parabolic stationary point is a maximum **\n");

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


	for (i=0;i<num_pw*num_states;i++) { approx_state[i] += opt_step*direction[i];}

	orthonormalise(num_pw,num_states,approx_state);

	// Apply the H to this state
	c_apply_H(num_pw,num_states,approx_state,H_kinetic,H_local,gradient);

	// Compute the new energy estimate
	*energy = 0.0;
	for (nb=0;nb<num_states;nb++) {
		offset = num_pw*nb;
		tmp_sum = 0.0;
		for (np=0;np<num_pw;np++) {
			// The complex dot-product a.b is conjg(a)*b
			// tmp_sum is just the real part, so we only compute that
			tmp_sum += creal(conj(approx_state[offset+np])*gradient[offset+np]);
		}
		eigenvalue[nb] = tmp_sum;
		*energy += tmp_sum;
	}

	// This ought to be the best, but check...
	if (*energy>best_energy) {
		// if(best_step>0.0_dp) then
		if (fabs(best_step-epsilon)>0.0) { // roughly machine epsilon in double precision

			for (i=0;i<num_pw*num_states;i++) {
				approx_state[i] += best_step*direction[i];
			}

			orthonormalise(num_pw,num_states,approx_state);

			// Apply the H to this state
			c_apply_H(num_pw,num_states,approx_state,H_kinetic,H_local,gradient);

			// Compute the new energy estimate
			*energy = 0.0;
			for (nb=0;nb<num_states;nb++) {
				offset = num_pw*nb;
				tmp_sum = 0.0;
				for (np=0;np<num_pw;np++) {
					// The complex dot-product a.b is conjg(a)*b
					// tmp_sum is just the real part, so we only compute that
					tmp_sum += creal(conj(approx_state[offset+np])*gradient[offset+np]);
				}
				eigenvalue[nb] = tmp_sum;
				*energy += tmp_sum;
			}
		} else {
			mpi_printf(world_rank,"Oh dear: %f\n",best_step);
			mpi_printf(world_rank,"Problem with line search\n");
			exit(EXIT_FAILURE);
		}
	}

	// We'll use this step as the basis of our trial step next time
	// trial_step = 2*opt_step;

	TRACEFREE(tmp_state);
	return;

}

void output_results(int num_pw, int num_states, double *H_local,
		fftw_complex *wvfn) {

	FILE* potential = fopen("pot.dat", "w");

	for (int i = 0; i < num_pw; ++i) {
		fprintf(potential, "%.12g\t%.12g\n", i / (double)(num_pw), H_local[i]);
	}

	fclose(potential);

	fftw_complex* realspace_wvfn = (fftw_complex*)TRACEFFTW_MALLOC(num_pw * sizeof(fftw_complex));

	fftw_plan plan = fftw_plan_dft_1d(num_pw, wvfn, realspace_wvfn, FFTW_FORWARD, FFTW_ESTIMATE);

	for (int nb = 0; nb < num_states; ++nb) {
		char filename[15];
		snprintf(filename, 15, "wvfn_%.4i.dat", nb);
		FILE* wvfn_file = fopen(filename, "w");

		fftw_execute_dft(plan, wvfn, realspace_wvfn);

		for (int np = 0; np < num_pw; ++np) {
			fprintf(wvfn_file, "%.12g\t%.12g\n", np / (double)(num_pw),
					creal(realspace_wvfn[np]));
		}

		fclose(wvfn_file);
	}

	fftw_destroy_plan(plan);

	TRACEFFTW_FREE(realspace_wvfn);
}
