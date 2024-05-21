#ifndef TC_SYCL_INTERFACES_H
#define TC_SYCL_INTERFACES_H

#include <complex>

// iterative solver routines
void sycl_iterative_solver(struct toycode_params *params,
		double *H_kinetic, double *H_local, double *nl_base_state,
		std::complex<double> *exact_state);
void sycl_randomise_state(int num_plane_waves, int num_states,
		std::complex<double> *trial_wvfn);
void sycl_take_exact_state(int num_plane_waves, int num_states,
		std::complex<double> *trial_wvfn, std::complex<double> *exact_state);
void sycl_orthonormalise(int num_plane_waves, int num_states,
		std::complex<double> *trial_wvfn);
void sycl_orthogonalise(int num_plane_waves, int num_states,
		std::complex<double> *state, std::complex<double> *ref_state);
void sycl_precondition(int num_plane_waves, int num_states,
		std::complex<double> *search_direction, std::complex<double> *trial_wvfn,
		double *H_kinetic);
void sycl_diagonalise(int num_plane_waves,int num_states, std::complex<double> *state,
		std::complex<double> *H_state, double *eigenvalues, std::complex<double> *rotation);
void sycl_transform(int num_plane_waves, int num_states, std::complex<double> *trial_wvfn,
		std::complex<double> *overlap);
void sycl_apply_hamiltonian(int num_plane_waves, int num_states,
		std::complex<double> *trial_wvfn, double *H_kinetic, double *H_local,
		double *nl_base_state, std::complex<double> *gradient);
void sycl_calculate_eigenvalues(int num_plane_waves, int num_states,
		std::complex<double> *state, std::complex<double> *gradient, double *eigenvalues);
void sycl_iterative_search(int num_plane_waves, int num_states, double *H_kinetic,
		double *H_local, double *nl_base_state, std::complex<double> *trial_wvfn,
		std::complex<double> *gradient, std::complex<double> *rotation, double *eigenvalues);
bool sycl_check_convergence(double previous_energy, double total_energy,
		double tolerance);

void sycl_line_search(int num_pw ,int num_states, std::complex<double> *approx_state,
		double *H_kinetic, double *H_local, double *nl_base_state,
		std::complex<double> *direction, std::complex<double> *gradient, double *eigenvalue, double *energy) ;

void sycl_fft_transpose(sycl::buffer<std::complex<double>> in,
		sycl::buffer<std::complex<double>> out,
		ptrdiff_t local_start, ptrdiff_t local_z, ptrdiff_t num_plane_waves,
		int direction);

#endif
