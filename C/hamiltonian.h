

// C interface to init_H
void c_init_H(int num_pw, double *H_kinetic, double *H_nonlocal);

// C interface to apply_H
void c_apply_H(int num_pw, int num_states, double *state, double *H_kinetic, double *H_nonlocal, double *H_state);

// C interface to compute_eigenvalues
void c_compute_eigenvalues(int num_pw,int num_states, double *state, double *H_kinetic, double *H_nonlocal, double *eigenvalues);

// C interface to construct_full_H
void c_construct_full_H(int num_pw, double *H_kinetic, double *H_nonlocal, double *full_H);

// C interface to init_random_seed
void c_init_random_seed();

// C interface to randomise_state()
void c_randomise_state(int num_pw,int num_states, double *state);

// C interface to line_search
//double c_line_search(int num_pw, int num_states, double *approx_state, double *H_kinetic, double *H_nonlocal, double *direction, double energy);
