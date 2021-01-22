#include <complex>
#include <vector>

inline double* complex_to_double(std::vector<std::complex<double>>& complex_array) {
  return &reinterpret_cast<double*>(complex_array.data())[0];
}

inline const double*
complex_to_double(const std::vector<std::complex<double>>& complex_array) {
  return &reinterpret_cast<const double*>(complex_array.data())[0];
}

// C interface to init_H
void c_init_H(std::vector<double>& H_kinetic,
              std::vector<double>& H_nonlocal);

// C interface to apply_H
void c_apply_H(int num_pw, int num_states, const std::vector<std::complex<double>>& state,
               const std::vector<double>& H_kinetic, const std::vector<double>& H_nonlocal,
               std::vector<std::complex<double>>& H_state);

// C interface to compute_eigenvalues
void c_compute_eigenvalues(int num_pw, int num_states, double* state, double* H_kinetic,
                           double* H_nonlocal, double* eigenvalues);

// C interface to construct_full_H
void c_construct_full_H(int num_pw, const std::vector<double>& H_kinetic,
                        const std::vector<double>& H_local,
                        std::vector<std::complex<double>>& full_H);

// C interface to init_random_seed
void c_init_random_seed();

// C interface to randomise_state()
void c_randomise_state(int num_pw, int num_states, std::vector<std::complex<double>>& state);

// C interface to line_search
// double c_line_search(int num_pw, int num_states, double *approx_state, double
// *H_kinetic, double *H_nonlocal, double *direction, double energy);
