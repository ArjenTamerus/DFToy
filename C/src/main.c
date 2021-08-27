/* 
 * Main.c
 *
 * Toy 'DFT-like' solver code.
 *
 */

#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include <complex.h>
#include <fftw3.h>
#include <math.h>
#include "parallel.h"
#include "interfaces.h"

int main(int argc, char **argv)
{
	double *H_kinetic, *H_local;
	fftw_complex *full_H = NULL;
	int num_wave_vectors, num_plane_waves;
	int num_pw_3d;
	int num_states;
	struct toycode_params params;

	// Right now, the only parallel bits are the ScaLAPACK/ELPA solvers
	init_parallel(argc, argv);

	get_configuration_params(argc, argv, &params);

	num_wave_vectors = params.num_wave_vectors;
	num_plane_waves = 2*num_wave_vectors+1;
	num_pw_3d = num_plane_waves*num_plane_waves*num_plane_waves;

	num_states = params.num_states;

	mpi_printf("num_wave_vectors:\t%d\n", num_wave_vectors);
	mpi_printf("num_plane_waves:\t%d\n", num_plane_waves);
	mpi_printf("num_pw_3d:\t\t%d\n", num_pw_3d);

	H_kinetic = calloc(num_pw_3d, sizeof(double));
	H_local = calloc(num_pw_3d, sizeof(double));

	if (par_root) {
		init_kinetic(H_kinetic, num_plane_waves);
		init_local(H_local, num_plane_waves);
	}

	if (params.run_exact_solver) {
		full_H = exact_solver(num_plane_waves, num_states, H_kinetic, H_local,
				params.exact_solver, params.keep_exact_solution); 
	}

	if (params.run_iterative_solver && par_root) {
		iterative_solver(num_plane_waves, num_states, H_kinetic, H_local, full_H);
	}

	if (full_H) {
		free(full_H);
	}

	if(par_root) {
		free(H_kinetic);
		free(H_local);
	}

	finalise_parallel();

	mpi_printf("Done.\n");
	
	return 0;
}

// Initialise Kinetic Energy part of the Hamiltonian (reciprocal space)
void init_kinetic(double *H_kinetic, int num_plane_waves)
{
	int x, y, z;
	int idx_, idy_, idz_;
	int pos;

	mpi_printf("Initialising kinetic Hamiltonian...\n");

	for(z = -(num_plane_waves/2); z < num_plane_waves/2+1; z++) {
		idz_ = z < 0 ? num_plane_waves+z : z;

		for(y = -(num_plane_waves/2); y < num_plane_waves/2+1; y++) {
			idy_ = y < 0 ? num_plane_waves+y : y;

			for(x = -(num_plane_waves/2); x < num_plane_waves/2+1; x++) {
				idx_ = x < 0 ? num_plane_waves+x : x;

				pos = idz_ * num_plane_waves * num_plane_waves + idy_ * num_plane_waves
					+ idx_;
				H_kinetic[pos] = 0.5*(x*x+y*y+z*z);
			}

		}

	}
}

// Initialise Local Energy part of the Hamiltonian (real space)
void init_local(double *H_local, int num_plane_waves)
{
	int x, y, z;
	int idx_, idy_, idz_;
	int pos;

	int num_pw_3d = num_plane_waves*num_plane_waves*num_plane_waves;

	mpi_printf("Initialising local Hamiltonian...\n");

	for(z = -(num_plane_waves/2); z < num_plane_waves/2+1; z++) {
		idz_ = z < 0 ? num_plane_waves+z : z;

		for(y = -(num_plane_waves/2); y < num_plane_waves/2+1; y++) {
			idy_ = y < 0 ? num_plane_waves+y : y;

			for(x = -(num_plane_waves/2); x < num_plane_waves/2+1; x++) {
				idx_ = x < 0 ? num_plane_waves+x : x;

				pos = idz_ * num_plane_waves * num_plane_waves + idy_ * num_plane_waves
					+ idx_;
				H_local[pos] = -0.37/(0.005 + fabs(sqrt(x*x+y*y+z*z)/num_pw_3d - 0.5));
			}
		}
	}

}
