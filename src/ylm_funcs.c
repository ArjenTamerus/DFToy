#include <math.h>
#include "interfaces.h"
#include "parallel.h"

void beta_apply_ylm(int n, int l, int m, double scale, int num_plane_waves,
		double *Vnl_base, double *beta) {
	double scale_factor = scale / sqrt(4*CONST_PI);
	int offset = n * num_plane_waves*num_plane_waves * num_plane_waves;

	if (l < 0 || l > 2) {
		mpi_error("beta_apply_ylm: l should be 0 <= 2");
	}

	if (abs(m) > l) {
		mpi_error("beta_apply_ylm: |m| should be <= l");
	}

	switch(l) {
		case 0:
			apply_ylm_0_0(scale_factor, num_plane_waves, Vnl_base,
 &(beta[offset]));
			break;
		case 1:
			switch(m) {
				case -1:
					apply_ylm_1_n1(scale_factor, num_plane_waves, Vnl_base,
 &(beta[offset]));
					break;
				case 0:
					apply_ylm_1_0(scale_factor, num_plane_waves, Vnl_base,
 &(beta[offset]));
					break;
				case 1:
					apply_ylm_1_1(scale_factor, num_plane_waves, Vnl_base,
 &(beta[offset]));
					break;
			};
			break;
		case 2:
			switch(m) {
				case -2:
					apply_ylm_2_n2(scale_factor, num_plane_waves, Vnl_base,
 &(beta[offset]));
					break;
				case -1:
					apply_ylm_2_n1(scale_factor, num_plane_waves, Vnl_base,
 &(beta[offset]));
					break;
				case 0:
					apply_ylm_2_0(scale_factor, num_plane_waves, Vnl_base,
 &(beta[offset]));
					break;
				case 1:
					apply_ylm_2_1(scale_factor, num_plane_waves, Vnl_base,
 &(beta[offset]));
					break;
				case 2:
					apply_ylm_2_2(scale_factor, num_plane_waves, Vnl_base,
 &(beta[offset]));
					break;
			};
			break;
		default:
			mpi_error("Unreachable");
			break;
	}

}

void apply_ylm_0_0(double scale_factor, int num_plane_waves, double *Vnl_base,
 double *beta)
{
	int x, y, z;
	int pos_x, pos_y, pos_z;
	int pos;

	int wavevectors = (num_plane_waves-1)/2;

	for(z = -wavevectors; z < wavevectors+1; z++) {
		pos_z = z < 0 ? num_plane_waves+z : z;
		for(y = -wavevectors; y < wavevectors+1; y++) {
			pos_y = y < 0 ? num_plane_waves+y : y;
			for(x = -wavevectors; x < wavevectors+1; x++) {
				pos_x = x < 0 ? num_plane_waves+x : x;
				
				pos = pos_z * num_plane_waves * num_plane_waves +
					pos_y * num_plane_waves + pos_x;

				beta[pos] = Vnl_base[pos] *  scale_factor;
			}
		}
	}
}

void apply_ylm_1_0(double scale_factor, int num_plane_waves, double *Vnl_base,
 double *beta)
{
	int x, y, z;
	int pos_x, pos_y, pos_z;
	int pos;

	int wavevectors = (num_plane_waves-1)/2;

	for(pos_z = 0; pos_z < num_plane_waves; pos_z++) {
		z = (pos_z-wavevectors) / wavevectors;

		for(pos_y = 0; pos_y < num_plane_waves; pos_y++) {

			for(pos_x = 0; pos_x < num_plane_waves; pos_x++) {
				
				pos = pos_z * num_plane_waves * num_plane_waves +
					pos_y * num_plane_waves + pos_x;

				beta[pos] = Vnl_base[pos] *  scale_factor*(sqrt(3.0)*z);
			}
		}
	}
}

void apply_ylm_1_1(double scale_factor, int num_plane_waves, double *Vnl_base,
 double *beta)
{
	int x, y, z;
	int pos_x, pos_y, pos_z;
	int pos;

	int wavevectors = (num_plane_waves-1)/2;

	for(pos_z = 0; pos_z < num_plane_waves; pos_z++) {
		z = (pos_z-wavevectors) / wavevectors;

		for(pos_y = 0; pos_y < num_plane_waves; pos_y++) {

			for(pos_x = 0; pos_x < num_plane_waves; pos_x++) {
				
				pos = pos_z * num_plane_waves * num_plane_waves +
					pos_y * num_plane_waves + pos_x;

				beta[pos] = Vnl_base[pos] *  -scale_factor*(sqrt(3.0)*y);
			}
		}
	}
}

void apply_ylm_1_n1(double scale_factor, int num_plane_waves, double *Vnl_base,
 double *beta)
{
	int x, y, z;
	int pos_x, pos_y, pos_z;
	int pos;

	int wavevectors = (num_plane_waves-1)/2;

	for(z = -wavevectors; z < wavevectors+1; z++) {
		pos_z = z < 0 ? num_plane_waves+z : z;
		for(y = -wavevectors; y < wavevectors+1; y++) {
			pos_y = y < 0 ? num_plane_waves+y : y;
			for(x = -wavevectors; x < wavevectors+1; x++) {
				pos_x = x < 0 ? num_plane_waves+x : x;
				
				pos = pos_z * num_plane_waves * num_plane_waves +
					pos_y * num_plane_waves + pos_x;

				beta[pos] = Vnl_base[pos] *  scale_factor*(sqrt(3.0)*x);
			}
		}
	}
}

void apply_ylm_2_0(double scale_factor, int num_plane_waves, double *Vnl_base,
 double *beta)
{
	int x, y, z;
	int pos_x, pos_y, pos_z;
	int pos;

	int wavevectors = (num_plane_waves-1)/2;

	for(pos_z = 0; pos_z < num_plane_waves; pos_z++) {
		z = (pos_z-wavevectors) / wavevectors;

		for(pos_y = 0; pos_y < num_plane_waves; pos_y++) {

			for(pos_x = 0; pos_x < num_plane_waves; pos_x++) {
				
				pos = pos_z * num_plane_waves * num_plane_waves +
					pos_y * num_plane_waves + pos_x;

				beta[pos] = Vnl_base[pos] *  scale_factor * ((sqrt(5.0)*(-1+3*z*z))/2);
			}
		}
	}
}

void apply_ylm_2_1(double scale_factor, int num_plane_waves, double *Vnl_base,
 double *beta)
{
	int x, y, z;
	int pos_x, pos_y, pos_z;
	int pos;

	int wavevectors = (num_plane_waves-1)/2;

	for(z = -wavevectors; z < wavevectors+1; z++) {
		pos_z = z < 0 ? num_plane_waves+z : z;
		for(y = -wavevectors; y < wavevectors+1; y++) {
			pos_y = y < 0 ? num_plane_waves+y : y;
			for(x = -wavevectors; x < wavevectors+1; x++) {
				pos_x = x < 0 ? num_plane_waves+x : x;
				
				pos = pos_z * num_plane_waves * num_plane_waves +
					pos_y * num_plane_waves + pos_x;

				beta[pos] = Vnl_base[pos] *  scale_factor * -(sqrt(15.0)*y*z);
			}
		}
	}
}

void apply_ylm_2_n1(double scale_factor, int num_plane_waves, double *Vnl_base,
 double *beta)
{
	int x, y, z;
	int pos_x, pos_y, pos_z;
	int pos;

	int wavevectors = (num_plane_waves-1)/2;

	for(z = -wavevectors; z < wavevectors+1; z++) {
		pos_z = z < 0 ? num_plane_waves+z : z;
		for(y = -wavevectors; y < wavevectors+1; y++) {
			pos_y = y < 0 ? num_plane_waves+y : y;
			for(x = -wavevectors; x < wavevectors+1; x++) {
				pos_x = x < 0 ? num_plane_waves+x : x;
				
				pos = pos_z * num_plane_waves * num_plane_waves +
					pos_y * num_plane_waves + pos_x;

				beta[pos] = Vnl_base[pos] *  scale_factor * (sqrt(15.0)*x*z);
			}
		}
	}
}

void apply_ylm_2_2(double scale_factor, int num_plane_waves, double *Vnl_base,
 double *beta)
{
	int x, y, z;
	int pos_x, pos_y, pos_z;
	int pos;

	int wavevectors = (num_plane_waves-1)/2;

	for(z = -wavevectors; z < wavevectors+1; z++) {
		pos_z = z < 0 ? num_plane_waves+z : z;
		for(y = -wavevectors; y < wavevectors+1; y++) {
			pos_y = y < 0 ? num_plane_waves+y : y;
			for(x = -wavevectors; x < wavevectors+1; x++) {
				pos_x = x < 0 ? num_plane_waves+x : x;
				
				pos = pos_z * num_plane_waves * num_plane_waves +
					pos_y * num_plane_waves + pos_x;

				beta[pos] = Vnl_base[pos] *  scale_factor * ((sqrt(15.0) * (-(x*x) + y*y))/2);
			}
		}
	}
}

void apply_ylm_2_n2(double scale_factor, int num_plane_waves, double *Vnl_base,
 double *beta)
{
	int x, y, z;
	int pos_x, pos_y, pos_z;
	int pos;

	int wavevectors = (num_plane_waves-1)/2;

	for(z = -wavevectors; z < wavevectors+1; z++) {
		pos_z = z < 0 ? num_plane_waves+z : z;
		for(y = -wavevectors; y < wavevectors+1; y++) {
			pos_y = y < 0 ? num_plane_waves+y : y;
			for(x = -wavevectors; x < wavevectors+1; x++) {
				pos_x = x < 0 ? num_plane_waves+x : x;
				
				pos = pos_z * num_plane_waves * num_plane_waves +
					pos_y * num_plane_waves + pos_x;

				beta[pos] = Vnl_base[pos] *  scale_factor * -(sqrt(15.0)*x*y);
			}
		}
	}
}

