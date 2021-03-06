/*
 * Exact.c
 *
 * Find eigenstates using exact diagonaliser.
 *
 */

#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include <complex.h>
#include <fftw3.h>
#include <math.h>
#include <lapacke.h>
#include "parallel.h"
#include "interfaces.h"
#include "diag.h"
#include "trace.h"

fftw_complex *exact_solver(int num_plane_waves, int num_states,
		double *H_kinetic, double *H_local, const char *exact_diagonaliser,
		bool save_exact_state)
{
	fftw_complex *full_H;
	double *eigenvalues;
	int num_plane_waves_3D;

	struct tc_timer exact_timer;

	num_plane_waves_3D = num_plane_waves * num_plane_waves * num_plane_waves;

	mpi_printf("Calculating exact state.\n");

	if (num_plane_waves >= 11) {
		mpi_printf(
				"(WW) The exact solver scales extremely slowly: O((w^3)^2 in\n"
				"(WW) size, O((w^3)^3) in compute. Don't run with w > 5 for\n"
				"(WW) tests, or w > 8 unless you are testing diagonalisation\n"
				"(WW) library performance and have plenty of RAM available.\n"
				);
	}

	full_H = calloc(num_plane_waves_3D * num_plane_waves_3D,
			sizeof(fftw_complex));
	eigenvalues = calloc(num_plane_waves_3D, sizeof(double));

	construct_hamiltonian(full_H, H_kinetic, H_local, num_plane_waves);

	exact_timer = create_timer("Exact diagonalisation");

	start_timer(&exact_timer);

	diagonalise_exact_solution(full_H, eigenvalues, num_plane_waves_3D,
			num_states, exact_diagonaliser);

	stop_timer(&exact_timer);

	report_eigenvalues(eigenvalues, num_states);
	report_timer(&exact_timer);
	destroy_timer(&exact_timer);

	free(eigenvalues);

	if (save_exact_state) {
		return full_H;
	}
	else {
		free(full_H);

		return NULL;
	}

}

void construct_hamiltonian(fftw_complex *full_H, double *H_kinetic,
		double *H_local, int num_plane_waves)
{
	fftw_complex *tmp_state_1, *tmp_state_2;
	fftw_plan plan_forward, plan_backward;

	int i, j;
	int num_pw_3d = num_plane_waves * num_plane_waves * num_plane_waves;

	mpi_printf("Constructing full Hamiltonian...\n");

	tmp_state_1 = calloc(num_pw_3d, sizeof(fftw_complex));
	tmp_state_2 = calloc(num_pw_3d, sizeof(fftw_complex));

	plan_forward = fftw_plan_dft_3d(num_plane_waves, num_plane_waves,
			num_plane_waves, tmp_state_1, tmp_state_2, FFTW_FORWARD, FFTW_ESTIMATE);
	plan_backward = fftw_plan_dft_3d(num_plane_waves, num_plane_waves,
			num_plane_waves, tmp_state_1, tmp_state_2, FFTW_BACKWARD, FFTW_ESTIMATE);

	for(i = 0; i < num_pw_3d; i++) {

		for(j = 0; j < num_pw_3d; j++) {
			tmp_state_1[j] = 0+0*I;
		}
		tmp_state_1[i] = 1+0*I;

		fftw_execute_dft(plan_forward, tmp_state_1, tmp_state_2);

		for(j = 0; j < num_pw_3d; j++) {
			tmp_state_2[j] *= H_local[j]/num_pw_3d;
		}

		fftw_execute_dft(plan_backward, tmp_state_2, tmp_state_1);

		for(j = 0; j < num_pw_3d; j++) {
			full_H[i*num_pw_3d+j] = tmp_state_1[j];
		}
		
	}

	for(i = 0; i < num_pw_3d; i++) {
		full_H[i*num_pw_3d+i] += H_kinetic[i]+0*I;
	}

	fftw_destroy_plan(plan_forward);
	fftw_destroy_plan(plan_backward);

	free(tmp_state_1);
	free(tmp_state_2);
}

void diagonalise_exact_solution(fftw_complex *full_H, double *eigenvalues,
		int num_plane_waves, int num_states, const char *exact_diagonaliser)
{
	int diag_mode = get_diag_mode(exact_diagonaliser);
	
	switch (diag_mode) {
		case 0:
			diag_zheev(full_H, eigenvalues, num_plane_waves);
			break;
		case 1:
			diag_zheevd(full_H, eigenvalues, num_plane_waves);
			break;
		case 2:
			diag_zheevr(full_H, eigenvalues, num_plane_waves, num_states);
			break;
#ifdef DFTOY_USE_SCALAPACK
		case 3:
			diag_pzheev(full_H, eigenvalues, num_plane_waves);
			break;
		case 4:
			diag_pzheevd(full_H, eigenvalues, num_plane_waves);
			break;
		case 5:
			diag_pzheevr(full_H, eigenvalues, num_plane_waves, num_states);
			break;
#endif
#ifdef DFTOY_USE_ELPA
		case 6:
			diag_elpa(full_H, eigenvalues, num_plane_waves);
			break;
#endif
		default:
			diag_zheev(full_H, eigenvalues, num_plane_waves);
			break;
	};
}

void report_eigenvalues(double *eigenvalues, int num_states)
{
	int i;

	mpi_printf("==========================\n");
	mpi_printf("== EIGENSTATES REPORT   ==\n");
	mpi_printf("==========================\n");
	for(i = 0; i < num_states; i++) {
		mpi_printf("== | %d | %f |\t==\n", i, eigenvalues[i]);
	}
	mpi_printf("==========================\n");
}
