#ifndef TC_INTERFACES_H
#define TC_INTERFACES_H

#include <stdbool.h>
#include <complex.h>
#include <fftw3.h>

#define SOLVER_MAX_LEN 24

// TODO move/fix
#define CONST_E 2.71828182845904523536
#define CONST_PI 3.1415
static const fftw_complex CMPLX_0 = 0.0+0.0*I;
static const fftw_complex CMPLX_1 = 1.0+0.0*I;

struct toycode_params
{
	long int num_wave_vectors;
	long int num_states;

	bool run_exact_solver;
	bool run_iterative_solver;
	bool keep_exact_solution;

	const char *exact_solver;
};

void init_kinetic(double *H_kin, int num_plane_waves);
void init_local(double *H_loc, int num_plane_waves);

// Exact solver routines
fftw_complex *exact_solver(int num_plane_waves, int num_states,
		double *H_kinetic, double *H_local, const char *exact_diagonaliser,
		bool save_exact_state);
void construct_hamiltonian(fftw_complex *full_H, double *H_kin, double *H_loc,
		int num_plane_waves);
void diagonalise_exact_solution(fftw_complex *full_H, double *eigenvalues,
		int num_plane_waves, int num_states_global, const char *exact_diagonaliser);
void report_eigenvalues(double *eigenvalues, int num_states);

// iterative solver routines
void iterative_solver(int num_plane_waves, int num_states,
		double *H_kinetic, double *H_local, double *nl_base_state,
		fftw_complex *exact_state);
void randomise_state(int num_plane_waves, int num_states,
		fftw_complex *trial_wvfn);
void take_exact_state(int num_plane_waves, int num_states,
		fftw_complex *trial_wvfn, fftw_complex *exact_state);
void orthonormalise(int num_plane_waves, int num_states,
		fftw_complex *trial_wvfn);
void orthogonalise(int num_plane_waves, int num_states,
		fftw_complex *state, fftw_complex *ref_state);
void precondition(int num_plane_waves, int num_states,
		fftw_complex *search_direction, fftw_complex *trial_wvfn,
		double *H_kinetic);
void diagonalise(int num_plane_waves,int num_states, fftw_complex *state,
		fftw_complex *H_state, double *eigenvalues, fftw_complex *rotation);
void transform(int num_plane_waves, int num_states, fftw_complex *trial_wvfn,
		fftw_complex *overlap);
void apply_hamiltonian(int num_plane_waves, int num_states,
		fftw_complex *trial_wvfn, double *H_kinetic, double *H_local,
		double *nl_base_state, fftw_complex *gradient);
void calculate_eigenvalues(int num_plane_waves, int num_states,
		fftw_complex *state, fftw_complex *gradient, double *eigenvalues);
void iterative_search(int num_plane_waves, int num_states, double *H_kinetic,
		double *H_local, double *nl_base_state, fftw_complex *trial_wvfn,
		fftw_complex *gradient, fftw_complex *rotation, double *eigenvalues);
bool check_convergence(double previous_energy, double total_energy,
		double tolerance);

void line_search(int num_pw ,int num_states, fftw_complex *approx_state,
		double *H_kinetic, double *H_local, double *nl_base_state,
		fftw_complex *direction, fftw_complex *gradient, double *eigenvalue, double *energy) ;

//support
void init_seed();
double random_double();


// parameter parsing
void set_default_configuration_params(struct toycode_params *params);
void get_configuration_params(int argc, char **argv,
		struct toycode_params *params);
void set_int_param(long int *param, const char *param_value,
		const char *param_name);
int get_diag_mode(const char *diag_param);
void version();
void usage();
const char *get_version_string();

// NEW
void init_nonlocal_base(double *nonlocal_base_state, int num_plane_waves);
void beta_apply_ylm(int n, int l, int m, double scale, int num_plane_waves,
		double *Vnl_base, double *beta);

// Ylm
void apply_ylm_0_0(double scale_factor, int num_plane_waves, double *Vnl_base,
 double *beta);
void apply_ylm_1_0(double scale_factor, int num_plane_waves, double *Vnl_base,
 double *beta);
void apply_ylm_1_1(double scale_factor, int num_plane_waves, double *Vnl_base,
 double *beta);
void apply_ylm_1_n1(double scale_factor, int num_plane_waves, double *Vnl_base,
 double *beta);
void apply_ylm_2_0(double scale_factor, int num_plane_waves, double *Vnl_base,
 double *beta);
void apply_ylm_2_1(double scale_factor, int num_plane_waves, double *Vnl_base,
 double *beta);
void apply_ylm_2_n1(double scale_factor, int num_plane_waves, double *Vnl_base,
 double *beta);
void apply_ylm_2_2(double scale_factor, int num_plane_waves, double *Vnl_base,
 double *beta);
void apply_ylm_2_n2(double scale_factor, int num_plane_waves, double *Vnl_base,
 double *beta);
#endif
