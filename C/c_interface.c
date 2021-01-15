#include <stdlib.h>


void init_h_(int *num_pw, double *H_kinetic, double *H_nonlocal);
void apply_h_(int *num_pw, int *num_states, double *state, double *H_kinetic, double *H_nonlocal, double *H_state);
//void compute_eigenvalues_(int *num_pw,int *num_states, double *state, double *H_kinetic, double *H_nonlocal, double *eigenvalues);
void construct_full_h_(int *num_pw, double *H_kinetic, double *H_nonlocal, double *full_H);
void randomise_state_(int *num_pw, int *num_states, double *state);
void line_search(int *num_pw,int *num_states, double *approx_state, double *H_kinetic, double *H_nonlocal, double *direction, double *gradient, double *eigenvalue, double *energy);
void init_random_seed_();


void c_init_H(int num_pw, double *H_kinetic, double *H_nonlocal) {

  // call the fortran routine
  init_h_(&num_pw,H_kinetic,H_nonlocal);
  return;

}

void c_apply_H(int num_pw, int num_states, double *state, double *H_kinetic, double *H_nonlocal, double *H_state) {

  // call the fortran routine
  apply_h_(&num_pw,&num_states,state,H_kinetic,H_nonlocal,H_state);
  return;

}

//void c_compute_eigenvalues(int num_pw,int num_states, double *state, double *H_kinetic, double *H_nonlocal, double *eigenvalues) {
//
//  // call the fortran routine
//  compute_eigenvalues_(&num_pw,&num_states,state,H_kinetic,H_nonlocal,eigenvalues);
//  return;
//
//}

void c_construct_full_H(int num_pw,double *H_kinetic, double *H_nonlocal, double *full_H) {

  // call the fortran routine
  construct_full_h_(&num_pw,H_kinetic,H_nonlocal,full_H);
  return;

}

void c_init_random_seed() {

    // call the fortran routine
    init_random_seed_();

}

void c_randomise_state(int num_pw, int num_states, double *state) {

    // call the fortran routine
    randomise_state_(&num_pw,&num_states,state);

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



