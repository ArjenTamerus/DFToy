#include "hamiltonian_c.hpp"

extern "C" {
void init_h_(const int* num_pw, double* H_kinetic, double* H_nonlocal);
void apply_h_(const int* num_pw, const int* num_states, const double* state,
              const double* H_kinetic, const double* H_nonlocal, double* H_state);
// void compute_eigenvalues_(int *num_pw,int *num_states, double *state, double
// *H_kinetic, double *H_nonlocal, double *eigenvalues);
void construct_full_h_(int* num_pw, const double* H_kinetic, const double* H_nonlocal,
                       double* full_H);
void randomise_state_(int* num_pw, int* num_states, double* state);
void line_search(int* num_pw, int* num_states, double* approx_state, double* H_kinetic,
                 double* H_nonlocal, double* direction, double* gradient,
                 double* eigenvalue, double* energy);
void init_random_seed_();
}

void c_init_H(std::vector<double>& H_kinetic,
              std::vector<double>& H_nonlocal) {

  const int num_pw = H_kinetic.size();
  // call the fortran routine
  init_h_(&num_pw, H_kinetic.data(), H_nonlocal.data());
  return;
}

void c_apply_H(int num_pw, int num_states, const std::vector<std::complex<double>>& state,
               const std::vector<double>& H_kinetic, const std::vector<double>& H_nonlocal,
               std::vector<std::complex<double>>& H_state) {

  // call the fortran routine
  apply_h_(&num_pw, &num_states, complex_to_double(state), H_kinetic.data(),
           H_nonlocal.data(), complex_to_double(H_state));
  return;
}

// void c_compute_eigenvalues(int num_pw,int num_states, double *state, double *H_kinetic,
// double *H_nonlocal, double *eigenvalues) {
//
//  // call the fortran routine
//  compute_eigenvalues_(&num_pw,&num_states,state,H_kinetic,H_nonlocal,eigenvalues);
//  return;
//
//}

void c_construct_full_H(int num_pw, const std::vector<double>& H_kinetic,
                        const std::vector<double>& H_local,
                        std::vector<std::complex<double>>& full_H) {
  // call the fortran routine
  construct_full_h_(&num_pw, H_kinetic.data(), H_local.data(), complex_to_double(full_H));
}

void c_init_random_seed() {

  // call the fortran routine
  init_random_seed_();
}

void c_randomise_state(int num_pw, int num_states, std::vector<std::complex<double>>& state) {

  // call the fortran routine
  randomise_state_(&num_pw, &num_states, complex_to_double(state));
}

/* double c_line_search (int num_pw, int num_states, double *approx_state, double
 * *H_kinetic, double *H_nonlocal, double *direction, double dumenergy) { */

/*     double *energy; */
/*     double renergy; */

/*     *energy = dumenergy; */
/*     // call the fortran routine  */
/*     line_search_(&num_pw,&num_states,approx_state,H_kinetic,H_nonlocal,direction,energy);
 */
/*     renergy = *energy; */
/*     return renergy; */

/* } */
