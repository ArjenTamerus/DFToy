#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include <complex.h>
#include <mpi.h>
#include <fftw3.h>
#include <math.h>
#include <lapacke.h>
#include "interfaces.h"

int main(int argc, char **argv)
{
	double *H_kinetic, *H_local;
	fftw_complex *full_H = NULL;
	int num_wave_vectors, num_plane_waves;
	int num_pw_3d;
	int num_states;
	//int diagonalisation_method;

	bool run_exact = true, run_iterative = true, keep_exact = false;

	num_wave_vectors = 3;
	num_plane_waves = 2*num_wave_vectors+1;
	num_pw_3d = num_plane_waves*num_plane_waves*num_plane_waves;

	num_states = 3;

	printf("num_wave_vectors:\t%d\n", num_wave_vectors);
	printf("num_plane_waves:\t%d\n", num_plane_waves);
	printf("num_pw_3d:\t\t%d\n", num_pw_3d);

	// Right now, MPI is only used for scaLAPACK routines
	MPI_Init(&argc, &argv);

	H_kinetic = calloc(num_pw_3d, sizeof(double));
	H_local = calloc(num_pw_3d, sizeof(double));

	init_kinetic(H_kinetic, num_plane_waves);
	init_local(H_local, num_plane_waves);

	if (run_exact) {
		full_H = exact_solver(num_plane_waves, num_states, H_kinetic, H_local,
				keep_exact); 
	}

	if (run_iterative) {
		printf("Calculating state iteratively.\n");

		iterative_solver(num_plane_waves, num_states, H_kinetic, H_local, full_H);
	}

	if (full_H) {
		free(full_H);
	}

	free(H_kinetic);
	free(H_local);

	MPI_Finalize();

	printf("Done.\n");
	
	return 0;
}

void init_kinetic(double *H_kinetic, int num_plane_waves)
{
	int x, y, z;
	int idx_, idy_, idz_;
	int pos;

	printf("Initialising kinetic Hamiltonian...\n");

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

void init_local(double *H_local, int num_plane_waves)
{
	int x, y, z;
	int idx_, idy_, idz_;
	int pos;

	int num_pw_3d = num_plane_waves*num_plane_waves*num_plane_waves;

	printf("Initialising local Hamiltonian...\n");

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
