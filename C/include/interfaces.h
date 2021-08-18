#include <fftw3.h>
void init_kinetic(double *H_kin, int num_plane_waves);
void init_local(double *H_loc, int num_plane_waves);
void construct_hamiltonian(fftw_complex *full_H, double *H_kin, double *H_loc, int num_plane_waves);
void diagonalise_exact_solution(fftw_complex *full_H, double *eigenvalues, int num_plane_waves);
void report_eigenvalues(double *eigenvalues, int num_states);

// iterative solver
void iterative_solver(int num_plane_waves, int num_states,
		double *H_kinetic, double *H_local, fftw_complex *exact_state);
void randomise_state(int num_plane_waves, int num_states, fftw_complex *trial_wvfn);
void take_exact_state(int num_plane_waves, int num_states,
		fftw_complex *trial_wvfn, fftw_complex *exact_state);
void orthonormalise(int num_plane_waves, int num_states, fftw_complex *trial_wvfn);
void orthogonalise(int num_plane_waves, int num_states, fftw_complex *state, fftw_complex *ref_state);
void precondition(int num_plane_waves, int num_states, fftw_complex *search_direction, fftw_complex *trial_wvfn, double *H_kinetic);
void diagonalise(int num_plane_waves,int num_states, fftw_complex *state, fftw_complex *H_state, double *eigenvalues, fftw_complex *rotation);
void transform(int num_plane_waves, int num_states, fftw_complex *trial_wvfn, fftw_complex *overlap);
void apply_hamiltonian(int num_plane_waves, int num_states, fftw_complex *trial_wvfn,
		double *H_kinetic, double *H_local, fftw_complex *gradient);
void calculate_eigenvalues(int num_plane_waves, int num_states,
		fftw_complex *state, fftw_complex *gradient, double *eigenvalues);
void iterative_search(int num_plane_waves, int num_states, double *H_kinetic,
		double *H_local, fftw_complex *trial_wvfn, fftw_complex *gradient,
		fftw_complex *rotation, double *eigenvalues);
int check_convergence(double previous_energy, double total_energy, double tolerance);

void line_search(int num_pw ,int num_states, fftw_complex *approx_state,
		double *H_kinetic, double *H_local, fftw_complex *direction,
		fftw_complex *gradient, double *eigenvalue, double *energy) ;
//support
void init_seed();
double random_double();
