/*
 * Iterative.c
 *
 * Find eigenstates using an iterative conjugate gradient approach.
 *
 */
#include <stdlib.h>
#include <stdio.h>
#include <complex.h>
#include <math.h>
#include <fftw3-mpi.h>
#include <cblas.h>
#include <lapacke.h>
#include "parallel.h"
#include "interfaces.h"
#include "trace.h"

static ptrdiff_t distr_npw_localsize, distr_local_npw, distr_local_start,
								 distr_local_n0, distr_max_n0;



void iterative_solver(struct toycode_params *params, double *global_H_kinetic,
		double *global_H_local, double *nl_base_state, fftw_complex *exact_state)
{
	fftw_complex *trial_wvfn,
							 *gradient,
							 *rotation;
	double *eigenvalues;

	int num_plane_waves = params->num_plane_waves;
	int num_states = params->num_states;
	int num_nl_states = params->num_nl_states;
	int num_pw_3d;

	struct tc_timer iterative_timer;

	num_pw_3d = num_plane_waves * num_plane_waves * num_plane_waves;

	// For now, set allocation size to be the same on each process.
	// Less thinking/logic in the manual transpose
	ptrdiff_t tmp_distr_npw_localsize = fftw_mpi_local_size_3d(num_plane_waves,
			num_plane_waves, num_plane_waves, MPI_COMM_WORLD, &distr_local_n0,
			&distr_local_start);

	MPI_Allreduce(&tmp_distr_npw_localsize, &distr_npw_localsize, 1, MPI_UINT64_T,
			MPI_MAX, MPI_COMM_WORLD);

	MPI_Allreduce(&distr_local_n0, &distr_max_n0, 1, MPI_UINT64_T,
			MPI_MAX, MPI_COMM_WORLD);

	distr_local_npw = distr_local_n0 * num_plane_waves * num_plane_waves;

	double *H_kinetic = &(global_H_kinetic[(distr_local_start)*num_plane_waves*num_plane_waves]);
	double *H_local = &(global_H_local[(distr_local_start)*num_plane_waves*num_plane_waves]);
	
	mpi_printf("Starting iterative solver\n");

	init_seed();

	trial_wvfn = calloc(distr_npw_localsize*num_states, sizeof(fftw_complex));
	gradient = calloc(distr_npw_localsize*num_states, sizeof(fftw_complex));
	rotation = calloc(num_states*num_states, sizeof(fftw_complex));

	eigenvalues = calloc(num_states, sizeof(double));

	// Initialisation and initial step

	if (exact_state != NULL) {
		// For verification - copying in the exact state should cause the iterative
		// solver to _immediately_ give the right answer. If it starts iterating or
		// the eigenstates don't match, there's a bug in the solver.
		mpi_printf("Using the exact solution as input for the iterative solver.\n"
				"This should give the correct answer immediately. [Debug feature]\n");
		take_exact_state(num_pw_3d, num_states, trial_wvfn, exact_state);
	}
	else {
		// Actual iterative method: start with random guess for wavefunction
		randomise_state(num_plane_waves, num_states, trial_wvfn);
	}

	iterative_timer = create_timer("Iterative diagonalisation");
	start_timer(&iterative_timer);

	orthonormalise(distr_npw_localsize, num_states, trial_wvfn);

	apply_hamiltonian(num_plane_waves, num_states, trial_wvfn, H_kinetic, H_local,
			nl_base_state, gradient);

	calculate_eigenvalues(distr_npw_localsize, num_states, trial_wvfn, gradient,
			eigenvalues);

	//// Iterate to find true eigenstates
	iterative_search(num_plane_waves, num_states, H_kinetic, H_local,
			nl_base_state, trial_wvfn, gradient, rotation, eigenvalues);

	// Rotate states to approach true eigenstates
	diagonalise(distr_npw_localsize,num_states,trial_wvfn,gradient,eigenvalues,rotation);

	stop_timer(&iterative_timer);

	report_eigenvalues(eigenvalues, num_states);

	report_timer(&iterative_timer);
	destroy_timer(&iterative_timer);

	free(trial_wvfn);
	free(gradient);
	free(rotation);
	free(eigenvalues);

}

// Guess a wavefunction: random but symmetric values (reciprocal space)
void randomise_state(int num_plane_waves, int num_states, fftw_complex *state)
{
	double rnd1, rnd2;
	int ns;
	int num_pw_2d = num_plane_waves * num_plane_waves;
	int num_pw_3d = num_plane_waves * num_plane_waves * num_plane_waves;

	mpi_printf("Randomise state\n");

	int x, y, z;
	int pos;
	int offset, offset_ns;
	fftw_complex rand_val;


	fftw_complex *tmp_state = calloc(num_pw_3d*num_states,sizeof(fftw_complex));

	// [TODO FIX] Current implementation: calculate the entire state on each MPI
	// process, then slice out our local state and discard the global state. This
	// is stupidly inefficient, but will do until (if) I figure out how to safely
	// pseudo-randomise the state based on a seed, in parallel.


// Parallelising this messes with the order of rand() -> let's keep it serial,
// yeah?
//#pragma omp parallel for default(none) shared(num_states,num_plane_waves,num_pw_2d,num_pw_3d,state) private(ns,z,y,x,rnd1,rnd2,rand_val,pos,offset,offset_ns)
	for (ns = 0; ns < num_states; ns++) {
		offset_ns = ns * num_pw_3d;

		for(z = 0; z < num_plane_waves; z++) {
			offset = offset_ns + z * num_plane_waves *num_plane_waves;

			for(y = 0; y < num_plane_waves/2+1; y++) {

				for(x = 0; x < num_plane_waves/2+1; x++) {

					rnd1 = random_double()-0.5;
					if (x == 0 && y == 0 && z == 0) {
						rnd2 = 0.0;
					}
					else {
						rnd2 = random_double()-0.5;
					}

					rand_val = 2*(rnd1 + rnd2*I);

					pos = y * num_plane_waves + x;
					tmp_state[offset+pos] = rand_val;

					pos = (y+1) * num_plane_waves - x - 1;
					tmp_state[offset+pos] = conj(rand_val);

					pos = num_pw_2d - ((y+1) * num_plane_waves - x);
					tmp_state[offset+pos] = -rand_val;

					pos = num_pw_2d - (y * num_plane_waves + x)-1;
					tmp_state[offset+pos] = -conj(rand_val);
				}

			}

		}

	}


	// Copy our local portion of the global state before discarding global
	for(ns = 0; ns < num_states; ns++) {
		for(z = 0; z < distr_local_n0; z++) {
			for(y = 0; y < num_plane_waves; y++) {
				for(x = 0; x < num_plane_waves; x++) {
					state[ns*distr_npw_localsize + z*num_plane_waves*num_plane_waves+y*num_plane_waves+x] =
						tmp_state[ns*num_pw_3d + (z+distr_local_start)*num_plane_waves*num_plane_waves+y*num_plane_waves+x];
				}
			}
		}
	}


	free(tmp_state);

}

// Debug function: copy exact solution into trial wavefunction
void take_exact_state(int num_plane_waves, int num_states,
		fftw_complex *trial_wvfn, fftw_complex *exact_state)
{
	int np;

	if (!exact_state) {
		mpi_error("Exact state undefined, did you run the exact solver?"
				"\nExiting with failed state.\n");
		exit(EXIT_FAILURE);
	}

	mpi_printf("Copying exact eigenstates\n");

	for (np = 0; np < num_states * num_plane_waves; np++) {
		trial_wvfn[np] = exact_state[np];
	}

}

void orthonormalise(int num_plane_waves, int num_states, fftw_complex
		*trial_wvfn)
{
	fftw_complex *overlap, *global_overlap;
	int ns1, ns2, pw;
	int offset_ns2, offset_ns1;
	int err;

	overlap = calloc(num_states*num_states, sizeof(fftw_complex));
	global_overlap = calloc(num_states*num_states, sizeof(fftw_complex));

	// overlap matrix
#pragma omp parallel for default(none) shared(num_states,distr_npw_localsize, \
		distr_local_npw,overlap,trial_wvfn) \
	private(ns1,ns2,offset_ns1,offset_ns2,pw)
	for (ns2 = 0; ns2 < num_states; ns2++) {
		offset_ns2 = ns2*distr_npw_localsize;
		for (ns1 = 0; ns1 < num_states; ns1++) {
			offset_ns1 = ns1*distr_npw_localsize;

			overlap[ns2*num_states+ns1] = 0.0+0.0*I;

//#pragma novector // Intel icx 2023.1 miscompiles this loop and throws around
//								 // NaNs and Infinities for e.g. -w 5 -s 1 with -O2+ and AVX2.
//								 // TODO Fix? Report? Ignore?
			for (pw = 0; pw < distr_local_npw; pw++) {
				overlap[ns2*num_states+ns1] += conj(trial_wvfn[offset_ns1+pw])
					* trial_wvfn[offset_ns2+pw];
			}
		}
	}

	MPI_Allreduce(overlap, global_overlap, num_states*num_states*2, MPI_DOUBLE,
			MPI_SUM, MPI_COMM_WORLD);

	// compute cholesky factorisation of overlap matrix
	err = LAPACKE_zpotrf(LAPACK_COL_MAJOR, 'U', num_states, global_overlap, num_states);
	if (err) {
		mpi_error("zpotrf failed: %d\n", err);
		exit(EXIT_FAILURE);
	}

	// invert
	err = LAPACKE_ztrtri(LAPACK_COL_MAJOR, 'U', 'N', num_states, global_overlap,
			num_states);
	if (err) {
		mpi_error("ztrtri failed: %d\n", err);
		exit(EXIT_FAILURE);
	}

	// Set lower triangle to zero - N.B. column-major
// (usually) tiny, no point in parallelising.
//#pragma omp parallel for default(none) shared(num_states,overlap) private(ns2,ns1)
	for (ns2 = 0; ns2 < num_states; ns2++) {

		for (ns1 = ns2+1; ns1 < num_states; ns1++) {
			global_overlap[ns2*num_states+ns1] = (0.0+0.0*I);
		}

	}

	// apply orthonormalisation to trial wvfn
	transform(num_plane_waves, num_states, trial_wvfn, global_overlap);

	free(overlap);
	free(global_overlap);
}

void orthogonalise(int num_plane_waves, int num_states, fftw_complex *state,
		fftw_complex *ref_state) {
	fftw_complex overlap,global_overlap;
	int ref_state_offset;
	int state_offset;
	int ns1;
	int ns2;
	int pw;

	int local_ref_state_offset;
	int local_state_offset;


	for (ns2=0;ns2<num_states;ns2++) {
		state_offset = ns2*num_plane_waves;
#pragma omp parallel default(none) shared(distr_local_npw,global_overlap,num_states, \
		num_plane_waves,ref_state,state,ns2,state_offset,overlap) \
		private(ns1,pw,local_ref_state_offset,ref_state_offset,local_state_offset)
			{
		for (ns1=0;ns1<num_states;ns1++) {
			ref_state_offset = ns1*num_plane_waves;

			overlap = (0+0*I);

			// Calculate overlap
			// Dot. Prod. = SUM_i(cplx_conj(a)_i*b_i)
#pragma omp for reduction(+:overlap)
			for (pw=0; pw < distr_local_npw; pw++) {
				local_ref_state_offset = ref_state_offset+pw;
				local_state_offset = state_offset+pw;

				overlap += conj(ref_state[local_ref_state_offset])
					* state[local_state_offset];

			}

#pragma omp single
			{
			MPI_Allreduce(&overlap, &global_overlap, 2, MPI_DOUBLE, MPI_SUM,
					MPI_COMM_WORLD);
			}

			// remove overlap from state
#pragma omp for
			for (pw=0; pw < distr_local_npw; pw++) {
				local_ref_state_offset = ref_state_offset+pw;
				local_state_offset = state_offset+pw;

				state[local_state_offset] -= global_overlap*ref_state[local_ref_state_offset];
			}
		}
			}//end parallel
	}

}

// Apply a kinetic energy-based preconditioner to the search direction.
// This should improve the conditioning of the eigenvalue search.
void precondition(int num_plane_waves, int num_states,
		fftw_complex *search_direction, fftw_complex *trial_wvfn, double *H_kinetic)
{
	int np,ns;
	int offset;
	double kinetic_eigenvalue;
	double x, tmp; 

	for (ns = 0;ns < num_states; ns++) {
		/* |---------------------------------------------------------------------|
			 | You need to compute the kinetic energy "eigenvalue" for state ns.   |
			 | We don't have the true wavefunction yet, but our best guess is in   |
			 | trial_wvfn so we estimate the kinetic energy Ek as:                 |
			 |                                                                     |
			 |     E_k = trial_wvfn^+ H_kinetic trial_wvfn                         |
			 |                                                                     |
			 | where "^+" means take the Hermitian conjugate (transpose it and     |
			 | take the complex conjugate of each element). H_kinetic is a         |
			 | diagonal matrix, so rather than store a num_plane_waves *           |
			 | num_plane_waves matrix with most elements being zero, we instead    |
			 | just store the num_plane_waves non-zero elements in a 1D array.     |
			 | Thus the nth element of the result of operating with the kinetic    |
			 | energy operator on the wavefunction is:                             |
			 |                                                                     |
			 |     (H_kinetic trial_wvfn)(n) = H_kinetic(n)*trial_wvfn(n)          |
			 |                                                                     |
			 |---------------------------------------------------------------------| */

		offset = ns*num_plane_waves;
		kinetic_eigenvalue = 0.0;
		double global_kinetic_eigenvalue = 0.0;

#pragma omp parallel default(none) shared(num_states,num_plane_waves,trial_wvfn,H_kinetic,search_direction,offset,kinetic_eigenvalue,global_kinetic_eigenvalue) private(np,x,tmp)
		{
#pragma omp for reduction(+:kinetic_eigenvalue)
		for (np = 0; np < distr_local_npw; np++) {
			kinetic_eigenvalue += H_kinetic[np]
				* creal(trial_wvfn[offset + np]) * creal(trial_wvfn[offset + np])
				+ H_kinetic[np] * cimag(trial_wvfn[offset + np])
				* cimag(trial_wvfn[offset + np]);
		}

#pragma omp single
		{
		MPI_Allreduce(&kinetic_eigenvalue, &global_kinetic_eigenvalue, 1,
				MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		}

#pragma omp for
		for (np = 0; np < distr_local_npw; np++) {
			/* |----------------------------------------------------------------|
				 | You need to compute and apply the preconditioning, using the   |
				 | estimate of trial_wvfn's kinetic energy computed above and the |
				 | kinetic energy associated with each plane-wave basis function  |
				 |----------------------------------------------------------------| */

			x = H_kinetic[np] / global_kinetic_eigenvalue;

			// apply f(x) - keeping in mind search_direction is complex
			// f(x) = (8+4x+2x^2+x^3)/(8+4x+2x^2+x^3+x^4)
			// = (8+4x+2x^2+x^3)/(8+4x+2x^2+x^3)*x^4)
			//
			// 8+4x+2x^2+x^3 = 8 + x * (4+ x * (2 + x ))
			tmp = 8.0 + x * (4.0 + x * (2.0 + x));

			search_direction[offset + np] *= tmp / (tmp + x*x*x*x);


		}

		// end parallel
		}
	}

}


void diagonalise(int num_plane_waves,int num_states, fftw_complex *state,
		fftw_complex *H_state, double *eigenvalues, fftw_complex *rotation)
{
	/* |-------------------------------------------------|
		 | This subroutine takes a set of states and       |
		 | H acting on those states, and transforms the    |
		 | states to diagonalise <state|H|state>.          |
		 |-------------------------------------------------| */
	int ns1, ns2;
	int offset_ns1, offset_ns2;
	int i;
	int err;

	fftw_complex local_rotation[num_states*num_states];

	// Compute the subspace H matrix and store in rotation array
#pragma omp parallel for default(none) shared(num_states,num_plane_waves,state,local_rotation,H_state,distr_local_npw) private(ns1,ns2,offset_ns1,offset_ns2,i)
	for (ns2=0;ns2<num_states;ns2++) {
		offset_ns2 = ns2*num_plane_waves;
		for (ns1=0;ns1<num_states;ns1++) {
			offset_ns1 = ns1*num_plane_waves;
			local_rotation[ns2*num_states+ns1] = (0.0+0.0*I);
			for (i=0;i<distr_local_npw;i++) {

				// The complex dot-product a.b is conjg(a)*b
				local_rotation[ns2*num_states+ns1] += 
					conj(state[offset_ns1+i])*H_state[offset_ns2+i];

			}
		}
	}

	MPI_Allreduce(local_rotation, rotation, 2*num_states*num_states, MPI_DOUBLE,
			MPI_SUM, MPI_COMM_WORLD);

	// Diagonalise to get eigenvectors and eigenvalues
	err = LAPACKE_zheev(LAPACK_ROW_MAJOR, 'V', 'U', num_states, rotation,
			num_states, eigenvalues);

	if(err) {
		mpi_fail("LAPACK error in diagonalise().\n");
	}

	// Finally apply the diagonalising rotation to state
	// (and also to H_state, to keep it consistent with state)
	transform(num_plane_waves,num_states,state,rotation);
	transform(num_plane_waves,num_states,H_state,rotation);

}

void transform(int num_plane_waves, int num_states, fftw_complex *state,
		fftw_complex *transformation)
{
	fftw_complex *new_state;
	int ns1, ns2, pw;
	int offset_ns2, offset_ns1;

	new_state = calloc(num_plane_waves * num_states, sizeof(fftw_complex));

#pragma omp parallel default(none) \
	shared(new_state,state,transformation,num_plane_waves,num_states,distr_local_npw) \
	private(ns1,ns2,offset_ns2,offset_ns1,pw)
{
	for(ns2 = 0; ns2 < num_states; ns2++) {
		offset_ns2 = ns2*num_plane_waves;

#pragma omp for collapse(2)
		for(ns1 = 0; ns1 < num_states; ns1++) {
			//offset_ns1 = ns1*num_plane_waves;

			for(pw = 0; pw < distr_local_npw; pw++) {
				new_state[ns1*num_plane_waves + pw] += state[offset_ns2 + pw]
					* transformation[ns1 * num_states + ns2];
			}

		}

	}

	// copyback
#pragma omp for
	for (pw = 0; pw < num_plane_waves * num_states; pw++) {
		state[pw] = new_state[pw];
	}
}//end omp parallel

	free(new_state);
}


void calc_beta_phi(double *beta, fftw_complex *state, fftw_complex *beta_phi,
		int num_states, int num_plane_waves) {
}

void apply_hamiltonian(int num_plane_waves, int num_states, fftw_complex *state,
		double *H_kinetic, double *H_local, double *nl_base_state,
		fftw_complex *H_state)
{
	fftw_plan plan_forward_1d = NULL, plan_backward_1d = NULL;
	fftw_plan plan_forward_2d = NULL, plan_backward_2d = NULL;
	fftw_plan plan_forward_3d = NULL, plan_backward_3d = NULL;
	fftw_complex *tmp_state = NULL, *tmp_state_in = NULL;

	int ns, np, x, y, z;
	int num_pw_3d = num_plane_waves * num_plane_waves * num_plane_waves;

	int n[] = {num_plane_waves, num_plane_waves};

	/* V_nl stuff */
	int num_nl_states = 0;
	int d_m = num_nl_states;
	int d_n = num_nl_states;

	double *beta = NULL;
	double *nl_d = NULL;
	double *d_beta = NULL;
	fftw_complex *beta_phi = NULL;

	nl_d = calloc(d_m*d_n,sizeof(double));
	beta = calloc(num_pw_3d*d_m, sizeof(double));
	beta_phi = calloc(num_pw_3d*d_m,sizeof(fftw_complex));
	d_beta = calloc(d_m*d_m, sizeof(fftw_complex));


	/* end V_nl stuff */

	plan_forward_1d = fftw_plan_many_dft(
			 1, &num_plane_waves, distr_local_n0*num_plane_waves,
			 tmp_state_in, &num_plane_waves,
			 1, num_plane_waves,
			 tmp_state, &num_plane_waves,
			 1, num_plane_waves,
			 FFTW_FORWARD, FFTW_ESTIMATE
			 );

	plan_forward_2d = fftw_plan_many_dft(
			 2, n, distr_local_n0,
			 tmp_state_in, n,
			 1, num_plane_waves*num_plane_waves,
			 tmp_state, n,
			 1, num_plane_waves*num_plane_waves,
			 FFTW_BACKWARD, FFTW_ESTIMATE
			 );

	plan_backward_1d = fftw_plan_many_dft(
			 1, &num_plane_waves, distr_local_n0*num_plane_waves,
			 tmp_state_in, &num_plane_waves,
			 1, num_plane_waves,
			 tmp_state, &num_plane_waves,
			 1, num_plane_waves,
			 FFTW_BACKWARD, FFTW_ESTIMATE
			 );

	plan_backward_2d = fftw_plan_many_dft(
			 2, n, distr_local_n0,
			 tmp_state_in, n,
			 1, num_plane_waves*num_plane_waves,
			 tmp_state, n,
			 1, num_plane_waves*num_plane_waves,
			 FFTW_BACKWARD, FFTW_ESTIMATE
			 );

	tmp_state = calloc(distr_npw_localsize,sizeof(fftw_complex));
	tmp_state_in = calloc(distr_npw_localsize,sizeof(fftw_complex));
	
	//
	// V_nl - Calculate Beta [plane waves * m] from base state (e^-r)
	// by applying a different spherical harmonic for each m.
	//
	// This is not at all correct for the final implementation, but it gives us
	// some numbers to work with.
	//

	// Just apply successive Ylm with nonsense scale factor for each Vnl state.
	// If (num states) > (3), keep using l=2, m=0. Will be updated once I
	// understand this better...
	int l = 0; int m = 0;
	for (int n = 0; n < num_nl_states; n++) {
		switch(n) {
			case 0:
				l = 0; m = 0;
				break;
			case 1:
				l = 1; m = -1;
				break;
			case 2:
				l = 1; m = 0;
				break;
			case 3:
				l = 1, m = 1;
				break;
			default:
				l = 2, m = 0;
				break;
		}

		// Need to decide on _some_ number for the scale factor...
		double scale = 1.0;
		beta_apply_ylm(n, l, m, scale, num_plane_waves, nl_base_state, beta);
	}

	// Fill up D matrix with nonsense numbers. Just need it to have the right
	// properties for now (real, symmetric, dominant(?))
	for (int n = 0; n < d_n; n++) {
		for (int m = 0; m <= n; m++) {
			nl_d[n*d_n+m] = (0.1*(double)(m+1)/(n+1))/d_n;
			nl_d[m*d_n+n] = (0.1*(double)(m+1)/(n+1))/d_n;
		}
	}



	// Apply H (= K + V_loc + V_nl) to each state/band
#pragma omp parallel default(none) shared(num_states,num_pw_3d,tmp_state,tmp_state_in, \
		plan_forward_1d, plan_forward_2d, plan_backward_1d, plan_backward_2d, distr_local_start, \
		distr_max_n0, distr_local_npw, num_plane_waves, H_local, H_kinetic, beta, beta_phi, \
		d_beta, nl_d, H_state, d_m, d_n, distr_npw_localsize, state) \
	private(np, m, n, ns,)
	for(ns = 0; ns < num_states; ns++) {

		//
		// Preparation for non-local potential.
		//
		// V_nl * Psi_G = Beta_Gn * SUM_m(D_nm * SUM_G'(Beta_G'm)) * Psi_G'
		//              = Beta * D * (Beta<dagger> * Psi)
		//
		// Beta and D have been calculated before this loop.
		//


		//
		// Now start 'assembling' the hamiltonian (really applying its constituent
		// parts in-place)
		//

		// Load wvfn at state/band ns into work array

#pragma omp for
		for(np = 0; np < distr_local_npw; np++) {
			tmp_state_in[np] = state[ns*distr_npw_localsize+np];
		}

#pragma omp single
		{
		// Convert from reciprocal to real space
		fftw_execute_dft(plan_forward_2d, tmp_state_in, tmp_state);
		transpose_for_fftw(tmp_state, tmp_state_in, distr_local_start,
				distr_max_n0, num_plane_waves, XZ);
		fftw_execute_dft(plan_forward_1d, tmp_state_in, tmp_state);
		}

		// Apply local H in real space
#pragma omp for
		for(np = 0; np < distr_local_npw; np++){
			tmp_state_in[np] = H_local[np] * tmp_state[np]/num_pw_3d;
		}

		// Calculate beta_phi, SUM_m(Beta<dagger>_G'm * Psi_G') ( note Phi == Psi )
#pragma omp for collapse(2)
		for (int m = 0; m < d_m; m++) {
			for(np = 0; np < distr_local_npw; np++) {
				beta_phi[m*distr_local_npw+np] = 
					beta[(m+1)*distr_local_npw-np-1] * tmp_state_in[np];
			}
		}

		// Calculate d_beta (D * beta_phi)
#pragma omp for collapse(3)
		for (int m = 0; m < d_m; m++) {
			for(int n = 0; n < d_n; n++) {
				d_beta[m*d_m+n] = 0.0+0.0*I;
				for(np = 0; np < distr_local_npw; np++) {
					d_beta[m*d_m+n] += nl_d[m*d_m+n] * beta_phi[m*distr_local_npw+np];
				}
			}
		}

		// Apply V_nl in real space
#pragma omp for collapse(3)
		for (int n = 0; n < d_n; n++) {
			for (int m = 0; m < d_m; m++) {
				for(np = 0; np < distr_local_npw; np++) {
					tmp_state_in[np] +=
						beta[n*distr_local_npw+np] * d_beta[n*d_n+m];
				}
			}
		}

		// Convert back to reciprocal space
#pragma omp single
		{
		fftw_execute_dft(plan_backward_1d, tmp_state_in, tmp_state);
		transpose_for_fftw(tmp_state, tmp_state_in, distr_local_start,
				distr_max_n0, num_plane_waves, XZ);
		fftw_execute_dft(plan_backward_2d, tmp_state_in, &H_state[ns*distr_npw_localsize]);
		}


		// Apply Kinetic H in reciprocal space
#pragma omp for
		for(np = 0; np < distr_local_npw; np++) {
			H_state[ns*distr_npw_localsize+np] = H_state[ns*distr_npw_localsize+np] 
				+ H_kinetic[np]*state[ns*distr_npw_localsize+np];
			
		}

	}

	free(beta);
	free(nl_d);
	free(d_beta);
	free(beta_phi);

	free(tmp_state);
	free(tmp_state_in);

	fftw_destroy_plan(plan_forward_1d);
	fftw_destroy_plan(plan_forward_2d);
	fftw_destroy_plan(plan_backward_1d);
	fftw_destroy_plan(plan_backward_2d);

}

double random_double()
{
	return (double)rand() / (double)RAND_MAX;
}

void init_seed()
{
	static int initialised = 0;
	int seed;

	if(!initialised) {
		// TODO enable _actually random_ seed?
		seed = 13377331;
		srand(seed);
		initialised = 1;
	}
}

void line_search(int num_plane_waves ,int num_states,
		fftw_complex *approx_state, double *H_kinetic, double *H_local,
		double *nl_base_state,
		fftw_complex *direction, fftw_complex *gradient, double *eigenvalue,
		double *energy)
{
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
	//double tmp_sum;
	int i,loop,ns,np,offset;
	double trial_step = 0.4;

	//int num_pw_3d = num_plane_waves * num_plane_waves * num_plane_waves;

	// C doesn't have a nice epsilon() function like Fortran, so
	// we use a lapack routine for this.
	epsilon = LAPACKE_dlamch('e');

	// To try to keep a convenient step length, we reduce the size of the search
	// direction
	mean_norm = 0.0;
	tmp_sum = 0.0;
//#pragma omp parallel for default(none) private(ns,offset,tmp_sum,np) \
//	shared(num_states,num_pw_3d,direction) reduction(+:mean_norm)
	for (ns = 0; ns < num_states; ns++) {
		offset=ns*distr_npw_localsize;

		for (np = 0; np < distr_local_npw; np++) {
			// NOTE apparently taking mean_norm as sum(abs(direction)) converged
			// faster than the original? Why?
			//mean_norm += cabs(direction[offset+np]);
			tmp_sum += pow(cabs(direction[offset+np]),2);
		}

		//tmp_sum    = sqrt(tmp_sum);
		//mean_norm += tmp_sum;
	}

	MPI_Allreduce(&tmp_sum, &mean_norm, 1, MPI_DOUBLE, MPI_SUM,
			MPI_COMM_WORLD);

	mean_norm     = mean_norm/(double)num_states;
	inv_mean_norm = 1.0/mean_norm;

	for (np = 0; np < num_states * distr_npw_localsize; np++) {
		direction[np] = direction[np]*inv_mean_norm;
	}

	// The rate-of-change of the energy is just 2*Re{direction.gradient}
	denergy_dstep = 0.0;

//#pragma omp parallel for default(none) \
//	private(ns,offset,tmp_sum,np) \
//	shared(num_states,num_pw_3d,direction,gradient) \
//	reduction(+:denergy_dstep)
	tmp_sum = 0.0;
	for ( ns =0; ns < num_states; ns++) {

		offset = distr_npw_localsize*ns;

		for (np = 0; np < distr_local_npw; np++) {
			// The complex dot-product is conjg(trial_wvfn)*gradient
			// tmp_sum is the real part, so we only compute that
			tmp_sum += creal(conj(direction[offset+np])*gradient[offset+np]);
		}

		//denergy_dstep += 2.0*tmp_sum;
	}

	tmp_sum *= 2;
	MPI_Allreduce(&tmp_sum, &denergy_dstep, 1, MPI_DOUBLE, MPI_SUM,
			MPI_COMM_WORLD);

	tmp_state = (fftw_complex *)calloc(distr_npw_localsize*num_states,sizeof(fftw_complex));

	best_step   = 0.0;
	best_energy = *energy;

	// First take a trial step in the direction
	step = trial_step;

	// We find a trial step that lowers the energy:
	for (loop = 0; loop < 10; loop++) {

//#pragma omp parallel for schedule(static,num_pw_3d) default(none) private(i) \
//		shared(num_states,num_pw_3d,tmp_state,approx_state,direction,step)
		for (int k = 0; k < num_states; k++) {
			for (int j = 0; j < distr_local_npw; j++) {
				i = k * distr_npw_localsize + j;
				tmp_state[i] = approx_state[i] + step*direction[i];
			}
		}

		orthonormalise(distr_npw_localsize,num_states,tmp_state);

		// Apply the H to this state
		apply_hamiltonian(num_plane_waves, num_states, tmp_state, H_kinetic,
				H_local, nl_base_state, gradient);

		// Compute the new energy estimate
		tmp_energy = 0.0;

//#pragma omp parallel for default(none) \
//		private(ns,offset,tmp_sum,np) \
//		shared(num_states,num_pw_3d,tmp_state,gradient) \
//		reduction(+:tmp_energy)
			tmp_sum = 0.0;
		for (ns = 0; ns < num_states; ns++) {

			offset = distr_npw_localsize*ns;

			for (np = 0; np < distr_local_npw; np++) {
				// The complex dot-product a.b is conjg(a)*b
				// tmp_sum is the real part, so we just compute that
				tmp_sum += creal(conj(tmp_state[offset+np])*gradient[offset+np]);
			}
			//tmp_energy += tmp_sum;
		}

		MPI_Allreduce(&tmp_sum, &tmp_energy, 1, MPI_DOUBLE, MPI_SUM,
				MPI_COMM_WORLD);

		if (tmp_energy < *energy) {
			break;
		}
		else {
			d2E_dstep2 = (tmp_energy - *energy - step*denergy_dstep ) / (step*step);

			if(d2E_dstep2 < 0.0) {

				// TODO confirm this is an impossible-to-reach condition (see break
				// condition above)
				//if(tmp_energy < *energy) {
				//	break;
				//}
				//else {
					step = step/4.0;
				//}
			}
			else {
				step  = -denergy_dstep/(2*d2E_dstep2);
			}

		}
	}

	if (tmp_energy < best_energy) {
		best_step   = step;
		best_energy = tmp_energy;
	}

	// We now have the initial eigenvalue, the initial gradient, and a trial step
	// -- we fit a parabola, and jump to the estimated minimum position
	// Set default step and energy
	d2E_dstep2 = (tmp_energy - *energy - step*denergy_dstep ) / (step*step);

	if (d2E_dstep2 < 0.0) {
		// Parabolic fit gives a maximum, so no good
		mpi_printf("** Warning, parabolic stationary point is a maximum **\n");

		if (tmp_energy < *energy) {
			opt_step = step;
		}
		else {
			opt_step = 0.1*step;
		}

	}
	else {
		opt_step  = -denergy_dstep / (2.0*d2E_dstep2);
	}


	//    e = e0 + de*x + c*x**2
	// => c = (e - e0 - de*x)/x**2
	// => min. at -de/(2c)
	//
	//    de/dx = de + 2*c*x


//#pragma omp parallel for schedule(static,num_pw_3d)
	for (i = 0; i < distr_npw_localsize*num_states; i++) {
		approx_state[i] += opt_step*direction[i];
	}

	orthonormalise(distr_npw_localsize,num_states,approx_state);

	// Apply the H to this state
	apply_hamiltonian(num_plane_waves,num_states,approx_state,H_kinetic,H_local,
			nl_base_state, gradient);

	// Compute the new energy estimate
	//*energy = 0.0;
	double loop_energy = 0.0;
	double local_eigenvalue[num_states];
////#pragma omp parallel for default(none) \
//	private(ns,offset,tmp_sum,np) \
//	shared(num_states,num_pw_3d,approx_state,gradient,eigenvalue) \
//	reduction(+:loop_energy)
	for ( ns = 0; ns < num_states; ns++) {

		offset = distr_npw_localsize*ns;
		tmp_sum = 0.0;

		for (np = 0; np < distr_local_npw; np++) {
			// The complex dot-product a.b is conjg(a)*b
			// tmp_sum is just the real part, so we only compute that
			tmp_sum += creal(conj(approx_state[offset+np])*gradient[offset+np]);
		}

		local_eigenvalue[ns] = tmp_sum;
		loop_energy += tmp_sum;
	}
	//*energy = loop_energy;
	
	MPI_Allreduce(local_eigenvalue, eigenvalue, num_states, MPI_DOUBLE, MPI_SUM,
			MPI_COMM_WORLD);
	MPI_Allreduce(&loop_energy, energy, 1, MPI_DOUBLE, MPI_SUM,
			MPI_COMM_WORLD);

	// This ought to be the best, but check...
	if (*energy > best_energy) {

		// Roughly machine epsilon in double precision
		if (fabs(best_step - epsilon) > 0.0) {
//#pragma omp parallel for schedule(static,num_pw_3d)
			for (i = 0; i < distr_npw_localsize*num_states; i++) {
				approx_state[i] += best_step*direction[i];
			}

			orthonormalise(distr_npw_localsize,num_states,approx_state);

			// Apply the H to this state
			apply_hamiltonian(num_plane_waves,num_states,approx_state,H_kinetic,
					H_local, nl_base_state, gradient);

			// Compute the new energy estimate
			//*energy = 0.0;
			loop_energy = 0.0;
//#pragma omp parallel for default(none) \
//	private(ns,offset,tmp_sum,np) \
//	shared(num_states,num_pw_3d,approx_state,gradient,eigenvalue) \
//	reduction(+:loop_energy)
			for (ns = 0; ns < num_states; ns++) {

				offset = distr_npw_localsize*ns;
				tmp_sum = 0.0;
				for (np = 0; np < distr_local_npw; np++) {
					// The complex dot-product a.b is conjg(a)*b
					// tmp_sum is just the real part, so we only compute that
					tmp_sum += creal(conj(approx_state[offset+np])*gradient[offset+np]);
				}

				local_eigenvalue[ns] = tmp_sum;
				loop_energy += tmp_sum;

			}
			//*energy = loop_energy;
			MPI_Allreduce(local_eigenvalue, eigenvalue, num_states, MPI_DOUBLE, MPI_SUM,
					MPI_COMM_WORLD);
			MPI_Allreduce(&loop_energy, energy, 1, MPI_DOUBLE, MPI_SUM,
					MPI_COMM_WORLD);
		}
		else {
			mpi_error("Problem with line search: best_step < 0 " "[%f]\n", best_step);
			MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
		}
	}

	// We'll use this step as the basis of our trial step next time
	trial_step = 2*opt_step;

	free(tmp_state);
	return;

}

void calculate_eigenvalues(int num_plane_waves, int num_states,
		fftw_complex *state, fftw_complex *gradient, double *eigenvalues)
{
	int offset = 0;
	int ns, pw;

	double local_eigenvalues[num_states];

//#pragma omp parallel for default(none) shared(eigenvalues,gradient,state,num_states,num_plane_waves) private(ns,offset,pw)
	for (ns = 0; ns < num_states; ns++) {
		offset = ns * num_plane_waves;

		local_eigenvalues[ns] = 0.0;

		for (pw = 0; pw < num_plane_waves; pw++) {
			// The complex dot-product is conjg(trial_wvfn)*gradient
			// NB the eigenvalue is the real part of the product, so only compute that
			local_eigenvalues[ns] += creal(conj(state[offset + pw])*gradient[offset + pw]);

		}

	}

	MPI_Allreduce(local_eigenvalues, eigenvalues, ns, MPI_DOUBLE, MPI_SUM,
			MPI_COMM_WORLD);

}

void iterative_search(int num_plane_waves, int num_states, double *H_kinetic,
		double *H_local, double *nl_base_state, fftw_complex *trial_wvfn,
		fftw_complex *gradient, fftw_complex *rotation, double *eigenvalues)
{
	int ns;
	int iter, max_iter;
	fftw_complex *search_direction,
							 *previous_search_direction;

	int i, reset_sd;
	int offset;
	int CG_RESET=5;
	double gamma, gTxg=0.0, gTxg_prev=1.0;
	double previous_energy;
	double energy_tolerance = 1.e-11;

	double total_energy = 0.;

	int num_pw_3d = num_plane_waves * num_plane_waves * num_plane_waves;

	search_direction = calloc(distr_npw_localsize*num_states, sizeof(fftw_complex));
	previous_search_direction = calloc(distr_npw_localsize*num_states,
			sizeof(fftw_complex));

	mpi_printf("Starting iterative search @ tolerance of %1.0e\n",
			energy_tolerance);

	// Energy is the sum of the eigenvalues.
	for (ns = 0; ns < num_states; ns++) {
		total_energy += eigenvalues[ns];
	}

	mpi_printf("+-----------+----------------+-----------------+\n");
	mpi_printf("|           |  Total energy  |  DeltaE         |\n");
	mpi_printf("+-----------+----------------+-----------------+\n");
	mpi_printf("|  Initial  | % #14.8g |                 |\n", total_energy);
	mpi_printf("+-----------+----------------+-----------------+\n");
	max_iter = 4000;

	/* ----------------------------------------------------
		 | Begin the iterative search for eigenvalues         |
		 ---------------------------------------------------- */

	for (iter = 1; iter <= max_iter; iter++) {

		previous_energy = total_energy; // book keeping

		// The constrained gradient is H.wvfn - (wvfn.H.wvfn)*wvfn
		// -- i.e. it is orthogonal to wvfn which we enforce by 
		// calling the routine below. Remember H.wvfn is already
		// stored as gradient. You need to complete this function
		orthogonalise(distr_npw_localsize,num_states,gradient,trial_wvfn);

		// The steepest descent search direction is minus the gradient
//#pragma omp parallel for schedule(static,num_pw_3d)
		for ( i = 0; i < distr_npw_localsize*num_states; i++) {
			// cannot copy into previous_search_direction *here* because
			// search_direction gets mangled inside the line_search function
			search_direction[i] = -gradient[i];
		}

		//* PRECON */
		precondition(distr_npw_localsize, num_states, search_direction, trial_wvfn,
				H_kinetic);
		orthogonalise(distr_npw_localsize, num_states, search_direction, trial_wvfn);

		//Always calculate gT*g even on reset iteration - will need as gTxg_prev in
		//next iteration
		gTxg = 0.0;

		double local_gTxg = 0.0;
//#pragma omp parallel for default(none) private(ns,offset,i) \
//		shared(num_states,num_pw_3d,search_direction,gradient) \
//		reduction(+:gTxg)
		for(ns = 0; ns < num_states; ns++) {
			offset = ns*distr_npw_localsize;
			for(i=0; i < distr_local_npw; i++) {
				local_gTxg += creal(conj(search_direction[offset+i])*gradient[offset+i]);
			}
		}

		MPI_Allreduce(&local_gTxg, &gTxg, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

		if (reset_sd != 0) {
			gamma = gTxg / gTxg_prev;

//#pragma omp parallel for default(none) \
//			private(ns, offset, i) \
//			shared(num_states,num_pw_3d,gamma,search_direction,previous_search_direction)
			for (ns = 0; ns < num_states; ns++) {
				offset = ns*distr_npw_localsize;

				for (i = 0;i < distr_local_npw; i++) {
					search_direction[offset+i] += gamma
						* previous_search_direction[offset+i];
				}

			}

			orthogonalise(distr_npw_localsize,num_states,search_direction,trial_wvfn);
		}

		gTxg_prev = gTxg;

		// Remember search direction
//#pragma omp parallel for schedule(static,num_pw_3d)
		for( i = 0; i < distr_npw_localsize*num_states; i++) {
			previous_search_direction[i] = search_direction[i];
		}

		// Search along this direction for the best approx. eigenvectors, i.e. the
		// lowest energy.
		line_search(num_plane_waves, num_states, trial_wvfn, H_kinetic, H_local,
				nl_base_state, search_direction, gradient, eigenvalues, &total_energy);

		if (check_convergence(previous_energy, total_energy, energy_tolerance)) {
			break;
		}

		if (iter%CG_RESET==0) reset_sd = 1;

		// Energy is the sum of the eigenvalues
		total_energy = 0.0;
		for (ns = 0; ns < num_states ; ns++) {
			total_energy += eigenvalues[ns];
		}

		mpi_printf("|     %4d  | % #14.8g |  % #14.8g |\n",iter,total_energy,
				previous_energy-total_energy); 

	}

	free(previous_search_direction);
	free(search_direction);
}

bool check_convergence(double previous_energy, double total_energy,
		double tolerance)
{
		if(fabs(previous_energy-total_energy)<tolerance) {
				mpi_printf("+-----------+----------------+-----------------+\n");
				mpi_printf("Eigenvalues converged\n");
				return true;
		}
		return false;
}
