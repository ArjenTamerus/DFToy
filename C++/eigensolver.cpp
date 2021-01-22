/*--------------------------------------------------!
  ! This program is designed to illustrate the use  !
  ! of numerical methods and optimised software     !
  ! libraries to solve large Hermitian eigenvalue   !
  ! problems where only the lowest few eigenstates  !
  ! are required. Such problems arise in many       !
  ! electronic structure calculations, where we     !
  ! solve a large Schroedinger equation for just    !
  ! enough states to accommodate the number of      !
  ! electrons.                                      !
  !                                                 !
  ! In this program we will use the special case    !
  ! where the Hamiltonian is real and symmetric.    !
  ! This is a good approximation for large systems. !
  !-------------------------------------------------!
  ! C version by Phil Hasnip (University of York)   !
  ! and Dave Quigley (University of Warwick)        !
  !-------------------------------------------------!
  ! Version 0.9, last modified 8th Sept. 2020       !
  !-------------------------------------------------! */

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>

#include <complex>
#include <iostream>
#include <numeric>
#include <vector>

#include <fftw3.h>

#include <hamiltonian_c.hpp>

// Global variables
double trial_step = 0.4; // step for line search
unsigned int seed;       // random number seed

// Function prototypes
void exact_diagonalisation(const std::vector<double>& H_kinetic,
                           const std::vector<double>& H_local,
                           std::vector<double>& full_eigenvalues);
void orthonormalise(int num_pw, int num_states, std::vector<std::complex<double>>& state);
void orthogonalise(int num_pw, int num_states, std::vector<std::complex<double>>& state,
                   std::vector<std::complex<double>>& ref_state);
void transform(int num_pw, int num_states, std::vector<std::complex<double>>& state,
               std::vector<std::complex<double>>& transformation);
void diagonalise(int num_pw, int num_states, std::vector<std::complex<double>>& state,
                 std::vector<std::complex<double>>& H_state,
                 std::vector<double>& eigenvalues);
void precondition(int num_pw, int num_states,
                  std::vector<std::complex<double>>& search_direction,
                  std::vector<std::complex<double>>& trial_wvfn,
                  std::vector<double>& H_kinetic);
void line_search(int num_pw, int num_states,
                 std::vector<std::complex<double>>& approx_state,
                 const std::vector<double>& H_kinetic,
                 const std::vector<double>& H_nonlocal,
                 std::vector<std::complex<double>>& direction,
                 std::vector<std::complex<double>>& gradient,
                 std::vector<double>& eigenvalue, double& energy);
void output_results(int num_pw, int num_states, const std::vector<double>& H_local,
                    const std::vector<std::complex<double>>& wvfn);

// The complex dot-product is conjg(x)*y
// This version does a single element
std::complex<double> dot_product(std::complex<double> x, std::complex<double> y) {
  return std::conj(x) * y;
}

// Complex dot product over a range
std::complex<double> dot_product(const std::complex<double>* x,
                                 const std::complex<double>* y,
                                 std::size_t len) {
  return std::transform_reduce(x, x + len, y,
                               std::complex<double>{0.0, 0.0}, std::plus<>{},
                               [](auto& lhs, auto& rhs) { return std::conj(lhs) * rhs; });
}

double dot_product_real(const std::complex<double>* x, const std::complex<double>* y,
                        std::size_t len) {
  return std::transform_reduce(x, x + len, y, std::complex<double>{0.0, 0.0},
                               std::plus<>{}, [](auto& lhs, auto& rhs) {
                                 return (lhs.real() * rhs.real())
                                        + (lhs.imag() * rhs.imag());
                               }).real();
}

extern "C" {
// LAPACK function prototypes
void zheev_(char* jobz, char* uplo, int* N, double* A, int* ldA, double* w, double* cwork,
            int* lwork, double* work, int* status);
void zpotrf_(char* uplo, int* N, double* A, int* lda, int* status);
void ztrtri_(char* uplo, char* jobz, int* N, double* A, int* lda, int* status);
double dlamch_(char* arg);
}

// Main program unit
int main() {
  /* ---------------------
     | Initialise system |
     --------------------*/

  // No. nonzero wavevectors "G" in our wavefunction expansion
  constexpr int num_wavevectors = 400;

  // No. plane-waves in our wavefunction expansion. One plane-wave has
  // wavevector 0, and for all the others there are plane-waves at +/- G
  constexpr int num_pw = 2 * num_wavevectors + 1;

  // No. eigenstates to compute
  constexpr int num_states = 2;

  // Catch any nonsensical combinations of parameters
  if (num_states >= num_pw) {
    std::cout << "Error, num_states must be less than num_pw\n";
    return EXIT_FAILURE;
  }

  // Set tolerance on the eigenvalue sum when using an iterative search.
  // The iterative search will stop when the change in the eigenvalue sum
  // per iteration is less than this tolerance.
  constexpr double energy_tol = 1.0e-10;

  // Initialise random number generator
  c_init_random_seed();

  // Initialise and build the Hamiltonian, comprising two terms: the kinetic energy,
  // which is a diagonal matrix in the plane-wave basis (Fourier space); and the local
  // potential energy, which is a diagonal matrix in real space (direct space).
  //
  // The Hamiltonian is built inside libhamiltonian by populating these two arrays.
  // NB these arrays are *real* (not complex)
  std::vector<double> H_kinetic(num_pw);
  std::vector<double> H_local(num_pw);
  c_init_H(H_kinetic, H_local);

  /* ----------------------------------------------------
     | Perform full diagonalisation using LAPACK.       |
     | You will need to complete the function called    |
     ---------------------------------------------------- */
  std::cout << "Starting full diagonalisation...\n\n";

  auto init_cpu_time = clock();
  std::vector<double> full_eigenvalues(num_pw);
  exact_diagonalisation(H_kinetic, H_local, full_eigenvalues);
  auto curr_cpu_time = clock();
  auto exact_cpu_time = static_cast<double>(curr_cpu_time - init_cpu_time)
                        / static_cast<double>(CLOCKS_PER_SEC);

  std::cout << " State         Eigenvalue\n";
  for (int nb = 0; nb < num_states; nb++) {
    printf("     %d % #19.10g\n", 1 + nb, full_eigenvalues[nb]);
  }

  // Energy is the sum of the eigenvalues of the occupied states
  const double exact_energy = std::accumulate(full_eigenvalues.begin(),
                                              full_eigenvalues.begin() + num_states, 0.0);
  std::cout << "Ground state energy: " << exact_energy << "\n";

  std::cout << "Full diagonalisation took " << exact_cpu_time << " secs\n\n";

  // Allocate memory for iterative eigenvector search. Each of the following
  // are stored in column-major (Fortran) order. Each column contains the
  // plane wave co-efficients for a single particle wavefunction and there
  // are num_states particles.
  //
  // NB these are *complex* so need 2 "doubles" per element (one real, one imaginary)
  std::vector<std::complex<double>> trial_wvfn(num_pw * num_states);
  std::vector<std::complex<double>> gradient(num_pw * num_states);

  // We have num_states eigenvalue estimates (real) and products (complex)
  std::vector<double> eigenvalue(num_states);

  /* ----------------------------------------------------
   | Initialise the iterative search for the lowest     |
   | num_states eigenvalues.                            |
   ---------------------------------------------------- */
  printf("Starting iterative search for eigenvalues\n\n");
  printf("+-----------+----------------+-----------------+\n");
  printf("| iteration |     energy     |  energy change  |\n");
  printf("+-----------+----------------+-----------------+\n");

  init_cpu_time = clock();

  // We start from a random guess for trial_wvfn
  // this routine is in libhamiltonian
  c_randomise_state(num_pw, num_states, trial_wvfn);

  // All the wavefunctions should be normalised and orthogonal to each other
  // at every iteration. We enforce this in the initial random state here.
  orthonormalise(num_pw, num_states, trial_wvfn);

  // Apply the H to this state, store the result H.wvfn in gradient. As yet this is
  // unconstrained, i.e. following this gradient will break orthonormality.
  c_apply_H(num_pw, num_states, trial_wvfn, H_kinetic, H_local, gradient);

  // Compute the eigenvalues, i.e. the Rayleigh quotient for each eigenpair
  // Note that we don't compute a denominator here because our trial states
  // are normalised.
  for (int nb = 0; nb < num_states; ++nb) {
    int offset = nb * num_pw;
    eigenvalue[nb] = dot_product_real(&trial_wvfn[offset], &gradient[offset], num_pw);
  }

  // Energy is the sum of the eigenvalues.
  double energy = std::accumulate(eigenvalue.begin(), eigenvalue.end(), 0.0);

  printf("|  Initial  | % 14.8g |                 |\n", energy);

  // In case of problems, we cap the total number of iterations
  constexpr int max_iter = 40000;

  /* ----------------------------------------------------
   | Begin the iterative search for eigenvalues         |
   ---------------------------------------------------- */

  double prev_energy; // energy at last cycle

  bool reset_sd = false;
  double cg_beta_old;

  std::vector<std::complex<double>> prev_search_direction(gradient.size());

  for (int iter = 1; iter <= max_iter; iter++) {

    prev_energy = energy; // book keeping

    // The constrained gradient is H.wvfn - (wvfn.H.wvfn)*wvfn
    // -- i.e. it is orthogonal to wvfn which we enforce by
    // calling the routine below. Remember H.wvfn is already
    // stored as gradient. You need to complete this function
    orthogonalise(num_pw, num_states, gradient, trial_wvfn);

    // The steepest descent search direction is minus the gradient
    std::vector<std::complex<double>> search_direction(gradient.size());

    for (std::size_t i = 0; i < gradient.size(); ++i) {
      search_direction[i] = -gradient[i];
    }

    // Any modifications to the search direction go here. e.g.
    // preconditioning, implementation of conjugate gradients etc.
    precondition(num_pw, num_states, search_direction, trial_wvfn, H_kinetic);
    orthogonalise(num_pw, num_states, search_direction, trial_wvfn);

    double cg_beta = 0.0;
    for (int nb = 0; nb < num_states; nb++) {
      const int offset = nb * num_pw;
      cg_beta += dot_product_real(&search_direction[offset], &gradient[offset], num_pw);
    }

    if (reset_sd) {
      const double cg_gamma = cg_beta / cg_beta_old;
      for (int nb = 0; nb < num_states; nb++) {
        const int offset = nb * num_pw;
        for (int i = 0; i < num_pw; i++) {
          // The complex dot-product is conjg(trial_wvfn)*gradient
          // we just want the real part here
          search_direction[offset + i] +=
              std::complex<double>{cg_gamma * prev_search_direction[offset + i].real(),
                                   cg_gamma * prev_search_direction[offset + i].imag()};
        }
      }

      orthogonalise(num_pw, num_states, search_direction, trial_wvfn);
    }

    cg_beta_old = cg_beta;

    prev_search_direction = search_direction;

    // Search along this direction for the best approx. eigenvectors, i.e. the lowest
    // energy.
    line_search(num_pw, num_states, trial_wvfn, H_kinetic, H_local, search_direction,
                gradient, eigenvalue, energy);

    // Check convergence
    if (std::fabs(prev_energy - energy) < energy_tol) {
      if (not reset_sd) {
        std::cout << "+-----------+----------------+-----------------+\n"
                  << "Eigenvalues converged\n";
        break;
      } else {
        reset_sd = false;
      }
    } else {
      reset_sd = true;
    }

    // Reset the CG every 5 steps, to prevent it stagnating
    if (iter % 5 == 0) {
      reset_sd = false;
    }

    // Energy is the sum of the eigenvalues
    energy = std::accumulate(eigenvalue.begin(), eigenvalue.end(), 0.0);
    printf("|     %4d  | % 14.8g |  % 14.8g |\n", iter, energy, prev_energy - energy);
  }

  curr_cpu_time = clock();
  auto iterative_cpu_time = static_cast<double>(curr_cpu_time - init_cpu_time)
                            / static_cast<double>(CLOCKS_PER_SEC);
  std::cout << "Iterative search took " << iterative_cpu_time << " secs\n\n";

  // If you have multiple states, you may get a linear combination of
  // them rather than the pure states. This can be fixed by computing
  // the Hamiltonian matrix in the basis of the trial states (from
  // trial_wvfn and gradient), and then diagonalising that matrix.
  //
  // In other words, we rotate the states to be as near as possible to
  // the true eigenstates
  //
  // This is done in the "diagonalise" routine
  diagonalise(num_pw, num_states, trial_wvfn, gradient, eigenvalue);

  // Finally summarise the results
  // Finally summarise the results - we renumber the states to start at 1
  std::cout << "=============== FINAL RESULTS ===============\n"
            << "State                     Eigenvalue         \n"
            << "                Iterative            Exact   \n";
  for (int nb = 0; nb < num_states; nb++) {
    printf("   %d   % #18.8g  % #18.8g \n", 1 + nb, eigenvalue[nb], full_eigenvalues[nb]);
  }

  printf("---------------------------------------------\n");
  printf("Energy % #18.8g  % #18.8g\n\n", energy, exact_energy);
  printf("---------------------------------------------\n");
  printf("Time taken (s) % 10.5g      % 10.5g\n", iterative_cpu_time, exact_cpu_time);
  printf("=============================================\n\n");

  output_results(num_pw, num_states, H_local, trial_wvfn);
}

// -- YOU WILL NEED TO FINISH THESE FUNCTIONS --

void exact_diagonalisation(const std::vector<double>& H_kinetic,
                           const std::vector<double>& H_local,
                           std::vector<double>& full_eigenvalues) {
  //-------------------------------------------------//
  // This subroutine takes a compact representation  //
  // of the matrix H, constructs the full H, and     //
  // diagonalises to get the whole eigenspectrum.    //
  //-------------------------------------------------//

  int num_pw = H_kinetic.size();

  // First we allocate memory for and construct the full Hamiltonian
  std::vector<std::complex<double>> full_H(num_pw * num_pw);

  // This routine is in libhamiltonian
  c_construct_full_H(num_pw, H_kinetic, H_local, full_H);

  // Use LAPACK to get eigenvalues and eigenvectors
  // NB H is real and symmetric (but not packed)

  std::vector<double> lapack_real_work(3 * num_pw - 2);

  auto lapack_lwork = 2 * num_pw - 1;
  std::vector<double> lapack_cmplx_work(2 * lapack_lwork);

  int status;
  char jobz;
  char uplo;

  for (int i = 0; i < 3 * num_pw - 2; i++) {
    lapack_real_work[i] = 0.0;
  }
  for (int i = 0; i < 2 * lapack_lwork; i++) {
    lapack_cmplx_work[i] = 0.0;
  }

  jobz = 'V';
  uplo = 'U';
  zheev_(&jobz, &uplo, &num_pw, complex_to_double(full_H), &num_pw,
         full_eigenvalues.data(), lapack_cmplx_work.data(), &lapack_lwork,
         lapack_real_work.data(), &status);
  if (status != 0) {
    std::cout << "Error with zheev in exact_diagonalisation\n";
    std::exit(EXIT_FAILURE);
  }
}

void orthogonalise(int num_pw, int num_states, std::vector<std::complex<double>>& state,
                   std::vector<std::complex<double>>& ref_state) {
  /* |-------------------------------------------------|
     | This subroutine takes a set of states and       |
     | orthogonalises them to a set of reference       |
     | states.                                         |
     |-------------------------------------------------| */

  /* |---------------------------------------------------------------------|
     | You need to:                                                        |
     |                                                                     |
     | Compute the overlap of state nb2 with ref_state nb1                 |
     | (use the generalised "dot product" for complex vectors)             |
     |                                                                     |
     | Remove the overlapping parts of ref_state nb1 from state nb2        |
     |---------------------------------------------------------------------| */

  for (int nb2 = 0; nb2 < num_states; nb2++) {
    int offset2 = nb2 * num_pw;
    for (int nb1 = 0; nb1 < num_states; nb1++) {
      int offset1 = nb1 * num_pw;
      auto overlap = dot_product(&ref_state[offset1], &state[offset2], num_pw);
      // Remove the overlapping component from state
      for (int np = 0; np < num_pw; np++) {
        state[offset2 + np] -= overlap * ref_state[offset1 + np];
      }
    }
  }
}

void precondition(int num_pw, int num_states,
                  std::vector<std::complex<double>>& search_direction,
                  std::vector<std::complex<double>>& trial_wvfn,
                  std::vector<double>& H_kinetic) {
  /* |-------------------------------------------------|
     | This subroutine takes a search direction and    |
     | applies a simple kinetic energy-based           |
     | preconditioner to improve the conditioning of   |
     | the eigenvalue search.                          |
     |-------------------------------------------------| */

  for (int nb = 0; nb < num_states; nb++) {
    /* |---------------------------------------------------------------------|
       | You need to compute the kinetic energy "eigenvalue" for state nb.   |
       | We don't have the true wavefunction yet, but our best guess is in   |
       | trial_wvfn so we estimate the kinetic energy Ek as:                 |
       |                                                                     |
       |     E_k = trial_wvfn^+ H_kinetic trial_wvfn                         |
       |                                                                     |
       | where "^+" means take the Hermitian conjugate (transpose it and     |
       | take the complex conjugate of each element). H_kinetic is a         |
       | diagonal matrix, so rather than store a num_pw x num_pw matrix with |
       | most elements being zero, we instead just store the num_pw non-zero |
       | elements in a 1D array. Thus the nth element of the result of       |
       | operating with the kinetic energy operator on the wavefunction is:  |
       |                                                                     |
       |     (H_kinetic trial_wvfn)(n) = H_kinetic(n)*trial_wvfn(n)          |
       |                                                                     |
       |---------------------------------------------------------------------| */
    const int offset = nb * num_pw;

    double kinetic_eigenvalue = 0.0;
    for (int np = 0; np < num_pw; np++) {
      // H_kinetic is real, only trial_wvfn is complex
      kinetic_eigenvalue += H_kinetic[np] * pow(trial_wvfn[offset + np].real(), 2);
      kinetic_eigenvalue += H_kinetic[np] * pow(trial_wvfn[offset + np].imag(), 2);
    }

    for (int np = 0; np < num_pw; ++np) {
      /* |---------------------------------------------------------------------|
         | You need to compute and apply the preconditioning, using the        |
         | estimate of trial_wvfn's kinetic energy computed above and the      |
         | kinetic energy associated with each plane-wave basis function       |
         |---------------------------------------------------------------------| */

      // Compute and apply the preconditioning (NB the preconditioner is real)
      double x = H_kinetic[np] / kinetic_eigenvalue;
      double temp = 8.0 + x * (4.0 + x * (2.0 + 1.0 * x));

      search_direction[offset + np] = std::complex<double>{
          search_direction[offset + np].real() * temp / (temp + pow(x, 4)),
          search_direction[offset + np].imag() * temp / (temp + pow(x, 4))};
      // std:: cout << np << " " << search_direction[offset + np].real() << " " << search_direction[offset + np].imag() << "\n";
    }
  }
}

void diagonalise(int num_pw, int num_states, std::vector<std::complex<double>>& state,
                 std::vector<std::complex<double>>& H_state,
                 std::vector<double>& eigenvalues) {
  /* |-------------------------------------------------|
     | This subroutine takes a set of states and       |
     | H acting on those states, and transforms the    |
     | states to diagonalise <state|H|state>.          |
     |-------------------------------------------------| */

  std::vector<std::complex<double>> rotation(num_states * num_states);

  // Compute the subspace H matrix and store in rotation array
  for (int nb2 = 0; nb2 < num_states; nb2++) {
    int offset2 = nb2 * num_pw;
    for (int nb1 = 0; nb1 < num_states; nb1++) {
      int offset1 = nb1 * num_pw;
      rotation[nb2 * num_states + nb1] = dot_product(&state[offset1], &H_state[offset2], num_pw);
    }
  }

  // Use LAPACK to get eigenvalues and eigenvectors
  // NB H is Hermitian (but not packed)

  // Allocate and zero work space
  std::vector<double> lapack_real_work(3 * num_pw - 2, 0.0);

  auto lapack_lwork = 2 * num_pw - 1;
  std::vector<double> lapack_cmplx_work(2 * lapack_lwork, 0.0);

  // Diagonalise to get eigenvectors and eigenvalues
  int status;
  char jobz = 'V';
  char uplo = 'U';
  zheev_(&jobz, &uplo, &num_states, complex_to_double(rotation), &num_states,
         eigenvalues.data(), lapack_cmplx_work.data(), &lapack_lwork,
         lapack_real_work.data(), &status);
  if (status != 0) {
    printf("Error with zheev in diagonalise\n");
    exit(EXIT_FAILURE);
  }

  // Finally apply the diagonalising rotation to state
  // (and also to H_state, to keep it consistent with state)
  transform(num_pw, num_states, state, rotation);
  transform(num_pw, num_states, H_state, rotation);
}

// -- THE FOLLOWING SUBROUTINES ARE ALREADY WRITTEN --
//       (you may wish to optimise them though)

void orthonormalise(int num_pw, int num_states,
                    std::vector<std::complex<double>>& state) {
  //-------------------------------------------------//
  // This subroutine takes a set of states and       //
  // orthonormalises them.                           //
  //-------------------------------------------------//
  int status;

  // Compute the overlap matrix (using 1D storage)
  std::vector<std::complex<double>> overlap(num_states * num_states);

  for (int nb2 = 0; nb2 < num_states; nb2++) {
    int offset2 = nb2 * num_pw;
    for (int nb1 = 0; nb1 < num_states; nb1++) {
      int offset1 = nb1 * num_pw;
      overlap[nb2 * num_states + nb1] = dot_product(&state[offset1], &state[offset2], num_pw);
    }
  }

  // Compute orthogonalising transformation

  // First compute Cholesky (U.U^H) factorisation of the overlap matrix
  char uplo = 'U';
  zpotrf_(&uplo, &num_states, complex_to_double(overlap), &num_states, &status);
  if (status != 0) {
    std::cout << "zpotrf failed in orthonormalise (status " << status << ")\n";
    exit(EXIT_FAILURE);
  }

  // invert this upper triangular matrix
  char jobz = 'N';
  ztrtri_(&uplo, &jobz, &num_states, complex_to_double(overlap), &num_states, &status);
  if (status != 0) {
    printf("ztrtri failed in orthonormalise\n");
    exit(EXIT_FAILURE);
  }

  // Set lower triangle to zero - N.B. column-major
  for (int nb2 = 0; nb2 < num_states; nb2++) {
    for (int nb1 = nb2 + 1; nb1 < num_states; nb1++) {
      overlap[nb2 * num_states + nb1] = {0.0, 0.0};
    }
  }

  // overlap array now contains the (upper triangular) orthonormalising transformation
  transform(num_pw, num_states, state, overlap);
}

void transform(int num_pw, int num_states, std::vector<std::complex<double>>& state,
               std::vector<std::complex<double>>& transformation) {
  //-------------------------------------------------//
  // This subroutine takes a set of states and       //
  // orthonormalises them.                           //
  //-------------------------------------------------//
  std::vector<std::complex<double>> new_state(num_pw * num_states, 0.0);

  // Apply transformation to state and H_state
  for (int nb2 = 0; nb2 < num_states; nb2++) {
    int offset2 = nb2 * num_pw;
    for (int nb1 = 0; nb1 < num_states; nb1++) {
      int offset1 = nb1 * num_pw;
      for (int np = 0; np < num_pw; np++) {
        new_state[offset1 + np] +=
            state[offset2 + np] * transformation[nb1 * num_states + nb2];
      }
    }
  }

  state = new_state;
}

void line_search(int num_pw, int num_states,
                 std::vector<std::complex<double>>& approx_state,
                 const std::vector<double>& H_kinetic, const std::vector<double>& H_local,
                 std::vector<std::complex<double>>& direction,
                 std::vector<std::complex<double>>& gradient,
                 std::vector<double>& eigenvalue, double& energy) {
  //-------------------------------------------------//
  // This subroutine takes an approximate eigenstate //
  // and searches along a direction to find an       //
  // improved approximation.                         //
  //-------------------------------------------------//

  double tmp_energy;
  double opt_step;
  double d2E_dstep2;

  // C doesn't have a nice epsilon() function like Fortran, so
  // we use a lapack routine for this.
  char arg = 'e';
  double epsilon = dlamch_(&arg);

  // To try to keep a convenient step length, we reduce the size of the search direction
  double mean_norm = 0.0;
  for (int nb = 0; nb < num_states; nb++) {
    int offset = nb * num_pw;
    double tmp_sum = 0.0;
    for (int np = 0; np < num_pw; np++) {
      tmp_sum += std::norm(direction[offset + np]);
    }
    mean_norm += std::sqrt(tmp_sum);
  }
  mean_norm = mean_norm / (double)num_states;
  const double inv_mean_norm = 1.0 / mean_norm;

  for (auto& dir : direction) {
    dir *= inv_mean_norm;
  }

  // The rate-of-change of the energy is just 2*direction.gradient
  double denergy_dstep = 0.0;
  for (int nb = 0; nb < num_states; nb++) {
    int offset = num_pw * nb;
    const double tmp_sum = dot_product_real(&direction[offset], &gradient[offset], num_pw);
    denergy_dstep += 2.0 * tmp_sum;
  }

  std::vector<std::complex<double>> tmp_state(approx_state.size());

  double best_step = 0.0;
  double best_energy = energy;

  // First take a trial step in the direction
  double step = trial_step;

  // We find a trial step that lowers the energy:
  for (int loop = 0; loop < 10; loop++) {

    for (std::size_t i = 0; i < tmp_state.size(); i++) {
      tmp_state[i] = approx_state[i] + step * direction[i];
    }

    orthonormalise(num_pw, num_states, tmp_state);

    // Apply the H to this state
    c_apply_H(num_pw, num_states, tmp_state, H_kinetic, H_local, gradient);

    // Compute the new energy estimate
    tmp_energy = 0.0;
    for (int nb = 0; nb < num_states; nb++) {
      int offset = num_pw * nb;
      tmp_energy += dot_product_real(&tmp_state[offset], &gradient[offset], num_pw);
    }

    if (tmp_energy < energy) {
      break;
    } else {
      d2E_dstep2 = (tmp_energy - energy - step * denergy_dstep) / (step * step);
      if (d2E_dstep2 < 0.0) {
        if (tmp_energy < energy) {
          break;
        } else {
          step = step / 4.0;
        }
      } else {
        step = -denergy_dstep / (2 * d2E_dstep2);
      }
    }
  }

  if (tmp_energy < best_energy) {
    best_step = step;
    best_energy = tmp_energy;
  }

  // We now have the initial eigenvalue, the initial gradient, and a trial step
  // -- we fit a parabola, and jump to the estimated minimum position
  // Set default step and energy
  d2E_dstep2 = (tmp_energy - energy - step * denergy_dstep) / (step * step);

  if (d2E_dstep2 < 0.0) {
    // Parabolic fit gives a maximum, so no good
    printf("** Warning, parabolic stationary point is a maximum **\n");

    if (tmp_energy < energy) {
      opt_step = step;
    } else {
      opt_step = 0.1 * step;
    }
  } else {
    opt_step = -denergy_dstep / (2.0 * d2E_dstep2);
  }

  //    e = e0 + de*x + c*x**2
  // => c = (e - e0 - de*x)/x**2
  // => min. at -de/(2c)
  //
  //    de/dx = de + 2*c*x

  for (int i = 0; i < num_pw * num_states; i++) {
    approx_state[i] += opt_step * direction[i];
  }

  orthonormalise(num_pw, num_states, approx_state);

  // Apply the H to this state
  c_apply_H(num_pw, num_states, approx_state, H_kinetic, H_local, gradient);

  // Compute the new energy estimate
  energy = 0.0;
  for (int nb = 0; nb < num_states; nb++) {
    int offset = num_pw * nb;
    const double tmp_sum = dot_product_real(&approx_state[offset], &gradient[offset], num_pw);
    eigenvalue[nb] = tmp_sum;
    energy += tmp_sum;
  }

  // This ought to be the best, but check...
  if (energy > best_energy) {
    // if(best_step>0.0_dp) then
    if (fabs(best_step - epsilon) > 0.0) { // roughly machine epsilon in double precision

      for (int i = 0; i < num_pw * num_states; i++) {
        approx_state[i] += best_step * direction[i];
      }

      orthonormalise(num_pw, num_states, approx_state);

      // Apply the H to this state
      c_apply_H(num_pw, num_states, approx_state, H_kinetic, H_local, gradient);

      // Compute the new energy estimate
      energy = 0.0;
      for (int nb = 0; nb < num_states; nb++) {
        int offset = num_pw * nb;
        const double tmp_sum = dot_product_real(&approx_state[offset], &gradient[offset], num_pw);
        eigenvalue[nb] = tmp_sum;
        energy += tmp_sum;
      }
    } else {
      printf("Oh dear: %f\n", best_step);
      printf("Problem with line search\n");
      exit(EXIT_FAILURE);
    }
  }

  //       printf(" %f %d %f <- test2\n",opt_step,step,energy);

  // We'll use this step as the basis of our trial step next time
  // trial_step = 2*opt_step;
}

void output_results(int num_pw, int num_states, const std::vector<double>& H_local,
                    const std::vector<std::complex<double>>& wvfn) {

  FILE* potential = fopen("pot.dat", "w");

  for (std::size_t i = 0; i < H_local.size(); ++i) {
    fprintf(potential, "%.12g\t%.12g\n", i / static_cast<double>(num_pw), H_local[i]);
  }

  fclose(potential);

  fftw_complex* realspace_wvfn =
      (fftw_complex*)fftw_malloc(num_pw * sizeof(fftw_complex));
  fftw_complex* tmp_wvfn = (fftw_complex*)fftw_malloc(num_pw * sizeof(fftw_complex));

  fftw_plan plan =
      fftw_plan_dft_1d(num_pw, tmp_wvfn, realspace_wvfn, FFTW_FORWARD, FFTW_ESTIMATE);

  for (int nb = 0; nb < num_states; ++nb) {
    char filename[15];
    snprintf(filename, 15, "wvfn_%i.dat", nb);
    FILE* wvfn_file = fopen(filename, "w");

    int offset = nb * num_pw;
    memcpy(tmp_wvfn, &wvfn[offset], num_states * sizeof(fftw_complex));

    fftw_execute_dft(plan, tmp_wvfn, realspace_wvfn);

    for (int np = 0; np < num_pw; ++np) {
      fprintf(wvfn_file, "%.12g\t%.12g\n", np / static_cast<double>(num_pw),
              realspace_wvfn[np][0]);
    }

    fclose(wvfn_file);
  }
  fftw_destroy_plan(plan);
  free(realspace_wvfn);
  free(tmp_wvfn);
}
