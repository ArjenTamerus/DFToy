#include <stdlib.h>
#include <complex.h>
#include <math.h>
#include <fftw3.h>
#include "interface.h"
#include "trace.h"

double *H_kinetic;        // the kinetic energy operator
double *H_local;          // the local potential operator

double trial_step = 0.4; // step for line search
unsigned int seed;       // random number seed

//void init_h_(int *num_pw, double *H_kinetic, double *H_nonlocal);
//void apply_h_(int *num_pw, int *num_states, double *state, double *H_kinetic, double *H_nonlocal, double *H_state);
////void compute_eigenvalues_(int *num_pw,int *num_states, double *state, double *H_kinetic, double *H_nonlocal, double *eigenvalues);
//void construct_full_h_(int *num_pw, double *H_kinetic, double *H_nonlocal, double *full_H);
//void randomise_state_(int *num_pw, int *num_states, double *state);
//void init_random_seed_();
double random_double();


void c_init_H(int num_pw, int num_wavevectors, double *H_kinetic, double *H_local) {
	int np, npx, npy, npz;
	double cutoff = 0.5*num_wavevectors*num_wavevectors;
	int ctr = 1;
	double mod_sq;

	num_pw = 2*num_wavevectors+1;
	//int wv_v_3 = num_wavevectors*num_wavevectors*num_wavevectors-1;
	int wv_v_2 = num_wavevectors*num_wavevectors;

	//H_kinetic[0] = 0.0;

	//for (np = 1; np < num_pw/2+1; np++) {
	//for (npz = -num_wavevectors+1; npz < num_wavevectors; npz++) {
	//	for (npy = -num_wavevectors+1; npy < num_wavevectors; npy++) {
	//		for (npx = -num_wavevectors+1; npx < num_wavevectors; npx++) {
	//for (npz = 0; npz < num_wavevectors; npz++) {
		for (npy = 0; npy <= num_wavevectors; npy++) {
			for (npx = 0; npx <= num_wavevectors; npx++) {
				mod_sq = 0.5*(npx*npx+npy*npy);//+npz*npz);
				if (npx == 0) {
					H_kinetic[npy * num_pw] = mod_sq;
					if (npy > 0) {
						H_kinetic[(num_pw-npy)*num_pw] = mod_sq;
					}
				}
				else {
					H_kinetic[npy * num_pw+npx] = mod_sq;
					H_kinetic[(npy+1) * num_pw-npx] = mod_sq;
					if (npy > 0) {
						H_kinetic[(num_pw-npy)*num_pw+npx] = mod_sq;
						H_kinetic[(num_pw-npy+1)*num_pw-npx] = mod_sq;
					}
				}
			}
		}
	//}
	
	printf("CTR: %d, NUM_PW:%d\n", ctr, num_pw);

	//num_pw = 25;
	printf("H_kinetic:\n");
	printf("%d\n",wv_v_2);
	for (npy = 0; npy < 2*num_wavevectors+1; npy++) {
		for (npx = 0; npx < 2*num_wavevectors+1; npx++) {
			printf("[%f]", H_kinetic[npy*num_pw+npx]);
		}
		printf("\n");
	}
	///for (npy = 0; npy <= num_wavevectors; npy++) {
	///	for (npx = 0; npx <= num_wavevectors; npx++) {
	///		//printf("%f|%f\n", H_kinetic[np], H_kinetic[num_pw-np]);
	///		printf("[%f]",H_kinetic[ctr++]);
	///		}
	///	printf("\n");
	///}

	for (np = 0; np < num_pw; np++) {
		H_local[np] = -0.37/(0.005+fabs(((1.0*np)/num_pw)-0.5));
		//H_local[np1*num_pw*num_pw+np2*num_pw+np3] = -0.37/(0.005+fabs(((1.0*np1)/num_pw)-0.5));
	}

	//printf("H_local:\n");
	//for (np = 0; np < num_pw; np++) {
	//	printf("%f\n", H_local[np]);
	//}

}

void c_apply_H(int num_pw, int num_states, fftw_complex *state, double *H_kinetic, double *H_local, fftw_complex *H_state) {

	int nb, np;

	int64_t rnorm = num_pw*num_pw;

	fftw_plan plan_forward, plan_backward;

	fftw_complex *tmp_state = NULL, *tmp_state_in = NULL;

	tmp_state = TRACEFFTW_MALLOC(num_pw*num_pw*sizeof(fftw_complex));
	tmp_state_in = TRACEFFTW_MALLOC(num_pw*num_pw*sizeof(fftw_complex));
	
	plan_forward = fftw_plan_dft_2d(num_pw, num_pw, tmp_state_in, tmp_state, FFTW_FORWARD, FFTW_ESTIMATE);
	plan_backward = fftw_plan_dft_2d(num_pw, num_pw, tmp_state_in, tmp_state, FFTW_BACKWARD, FFTW_ESTIMATE);

	for(nb = 0; nb < num_states; nb++) {
		for(np = 0; np < num_pw*num_pw; np++) {
			tmp_state_in[np] = state[nb*num_pw*num_pw+np];
		}

		fftw_execute_dft(plan_forward, tmp_state_in, tmp_state);

		for(np = 0; np < num_pw*num_pw; np++){
			tmp_state[np] = H_local[np]*tmp_state[np]/rnorm;
		}

		fftw_execute_dft(plan_backward, tmp_state, &H_state[nb*num_pw*num_pw]);

		for(np = 0; np < num_pw*num_pw; np++) {
			H_state[nb*num_pw*num_pw+np] = H_state[nb*num_pw*num_pw+np] + H_kinetic[np]*state[nb*num_pw*num_pw+np];
		}
	}

	fftw_destroy_plan(plan_forward);
	fftw_destroy_plan(plan_backward);


	TRACEFFTW_FREE(tmp_state);
	TRACEFFTW_FREE(tmp_state_in);

}

//void c_compute_eigenvalues(int num_pw,int num_states, double *state, double *H_kinetic, double *H_nonlocal, double *eigenvalues) {
//
//  // call the fortran routine
//  compute_eigenvalues_(&num_pw,&num_states,state,H_kinetic,H_nonlocal,eigenvalues);
//  return;
//
//}

void construct_full_H(int num_pw,double *H_kinetic, double *H_nonlocal, fftw_complex *full_H) {
	fftw_complex *tmp_state1, *tmp_state2;
	fftw_plan plan_forward, plan_backward;
	int np1, np2;

	//memset(full_H, num_pw*num_pw*sizeof(fftw_complex), 0);	
	for (np1 = 0; np1 < num_pw*num_pw; np1++) {
		full_H[np1] = (0+0*I);
	}

	tmp_state1 = TRACEFFTW_MALLOC(num_pw*sizeof(fftw_complex));
	tmp_state2 = TRACEFFTW_MALLOC(num_pw*sizeof(fftw_complex));
	//memset(tmp_state2, num_pw*sizeof(fftw_complex), 0);	
	for (np1 = 0; np1 < num_pw; np1++) {
		tmp_state2[np1] = (0+0*I);
	}

	plan_forward = fftw_plan_dft_1d(num_pw, tmp_state1, tmp_state2, FFTW_FORWARD, FFTW_ESTIMATE);
	plan_backward = fftw_plan_dft_1d(num_pw, tmp_state1, tmp_state2, FFTW_BACKWARD, FFTW_ESTIMATE);

	for(np1 = 0; np1 < num_pw; np1++) {
		//memset(tmp_state1, num_pw*sizeof(fftw_complex), 0);	
		for (np2 = 0; np2 < num_pw; np2++) {
			tmp_state1[np2] = (0+0*I);
		}
		tmp_state1[np1] = (1+0*I);

		fftw_execute_dft(plan_forward, tmp_state1, tmp_state2);

		for(np2 = 0; np2 < num_pw; np2++)
		{
			tmp_state2[np2] = H_local[np2]*tmp_state2[np2]/(((double)1.0)*num_pw);
		}

		fftw_execute_dft(plan_backward, tmp_state2, tmp_state1);

		for(np2 = 0; np2 < num_pw; np2++)
		{
			full_H[np1*num_pw + np2] = tmp_state1[np2];
		}
	}

	fftw_destroy_plan(plan_forward);
	fftw_destroy_plan(plan_backward);

	TRACEFFTW_FREE(tmp_state1);
	TRACEFFTW_FREE(tmp_state2);

	// Add contribution from kinetic term
	for(int i=1;i<num_pw;i++) {
		full_H[i*num_pw+i] += H_kinetic[i];
	}

}

double random_double() {
	return (double)rand() / (double)RAND_MAX;
}

void c_init_random_seed() {
	static int initialised = 0;
	int seed;

	if(!initialised) {
		// TODO use clock
		seed = 13377331;
		srand(seed);
		initialised = 1;
	}
}

//void c_randomise_state(int num_pw, int num_states, fftw_complex state[num_states][num_pw])
void c_randomise_state(int num_pw, int num_states, fftw_complex *state)
{
	double rnd1, rnd2;
	int nb, npx, npy, npz;
	//int ctr;

	int num_wavevectors = num_pw/2; // woot for floor()
	fftw_complex rand_cmplx;

	for (nb = 0; nb < num_states; nb++) {
		//rnd1 = random_double();
		rand_cmplx = (random_double()+0.0*I);
		//ctr = 1;

		for (npy = 0; npy <= num_wavevectors; npy++) {
			for (npx = 0; npx <= num_wavevectors; npx++) {
				//rand_cmplx = 0.5*(npx*npx+npy*npy);//+npz*npz);
				if (npx == 0) {
					state[nb*num_pw*num_pw+npy * num_pw] = rand_cmplx;
					if (npy > 0) {
						state[nb*num_pw*num_pw+(num_pw-npy)*num_pw] = -rand_cmplx;
					}
				}
				else {
					state[nb*num_pw*num_pw+npy * num_pw+npx] = rand_cmplx;
					state[nb*num_pw*num_pw+(npy+1) * num_pw-npx] = conj(rand_cmplx);
					if (npy > 0) {
						state[nb*num_pw*num_pw+(num_pw-npy)*num_pw+npx] = -rand_cmplx;
						state[nb*num_pw*num_pw+(num_pw-npy+1)*num_pw-npx] = -conj(rand_cmplx);
					}
				}
				rand_cmplx = (random_double()+random_double()*I);
			}
		}

//		//for (npz = -num_wavevectors; npz <= num_wavevectors; npz++) {
//			for (npy = 0; npy < num_pw; npy++) {
//				for (npx = 0; npx < num_pw; npx++) {
//					//if (npz== 0 && npy== 0 && npx==0) continue;
//					rnd1 = random_double();
//					rnd2 = random_double();
//					state[nb*num_pw+ctr] = 2*((rnd1-0.5)+(rnd2-0.5)*I);
//					//cmplx conjg
//					//state[nb*num_pw*num_pw*num_pw+((num_pw-npz)*num_pw*num_pw)+(num_pw-npy)*num_pw-npx] = 2*((rnd1-0.5)+(rnd2-0.5)*I);
//					//state[(nb+1)*num_pw-np] = 2*((rnd1-0.5)-2*(rnd2-0.5)*I);
//				}
//			}
//		//}
	}

}

/* double c_line_search (int num_pw, int num_states, double *approx_state, double *H_kinetic, double *H_nonlocal, double *direction, double dumenergy) { */


/*     double *energy; */
/*     double renergy; */

/*     *energy = dumenergy; */
/*     // call the fortran routine  */
/*     line_search_(&num_pw,&num_states,approx_state,H_kinetic,H_nonlocal,direction,energy); */
/*     renergy = *energy; */
/*     return renergy; */

/* } */



