/*
 * Params.c
 *
 * Command line parameter processing.
 *
 */

#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <getopt.h>
#include <math.h>
#include "parallel.h"
#include "interfaces.h"

struct option tc_params[] = {
	{"cutoff", required_argument, NULL, 'c'},
	{"wavevectors", required_argument, 0, 'w'},
	{"states", required_argument, 0, 's'},
	{"bands", required_argument, 0, 'b'}, // == states
	{"nl_states", required_argument, 0, 'n'}, // non-local state
	{"exact_solver", required_argument, 0, 'x'},
	{"debug_iterative", no_argument, 0, 'd'},
	{"exact", no_argument, 0, 'e'},
	{"iterative", no_argument, 0, 'i'},
	{"help", no_argument, 0, 'h'},
	{"usage", no_argument, 0, 'u'},
	{"version", no_argument, 0, 'V'},
	{NULL, 0, NULL, 0}
};

void set_default_configuration_params(struct toycode_params *params)
{
	params->num_wave_vectors = 12;
	params->num_states = 4;
	params->num_nl_states = 1;

	params->cutoff_energy = 1.0;
	params->grid_scale = 1.75;

	// non-compute params
	params->kpoints = 1;
	params->spins = 1;
	params->is_gamma = false;
	
	// settings
	params->run_exact_solver = false;
	params->run_iterative_solver = true;
	params->keep_exact_solution = false;

	params->exact_solver = NULL;

	// Use GPU? [future]
	params->use_accelerator = false;
}

void set_double_param(double *param, const char *param_value,
		const char *param_name)
{
	char *endptr = NULL;

	*param = strtod(param_value, &endptr);

	if (*param <= 0.0 || *endptr != '\0') {
		mpi_error("Invalid value for parameter %s: %s (should be floating point)\n",
				param_name, param_value);
		mpi_fail("Exiting\n");
	}
}

void set_int_param(long int *param, const char *param_value,
		const char *param_name)
{
	char *endptr = NULL;

	*param = strtol(param_value, &endptr, 10);

	if (*param < 1 || *endptr != '\0') {
		mpi_error("Invalid value for parameter %s: %s (should be integer > 0)\n",
				param_name, param_value);
		mpi_fail("Exiting\n");
	}
}

void get_configuration_params(int argc, char **argv,
		struct toycode_params *params)
{
	int opt_id;
	bool set_debug_iterative = false;
	bool set_run_exact = false;
	bool set_run_iterative = false;

	set_default_configuration_params(params);


	// Use getopt to parse command line options
	while (1) {
		opt_id = getopt_long_only(argc, argv, "dehis:uVw:x:", tc_params, NULL);

		if(opt_id == -1) {
			break;
		}
		switch(opt_id) {
			case 'b':
				set_int_param(&(params->num_states), optarg, "--bands [-b]");
				break;

			case 'c':
				set_double_param(&(params->cutoff_energy), optarg, "--cutoff [-c]");
				break;

			case 'd':
				set_debug_iterative = true;
				break;

			case 'e':
				set_run_exact = true;
				break;

			case 'h':
			case 'u':
				usage();
				break;

			case 'i':
				set_run_iterative = true;
				break;

			case 'k':
				set_int_param(&(params->kpoints), optarg, "--kpoints [-k]");
				break;

			case 'n':
				set_int_param(&(params->num_nl_states), optarg, "--nl_states [-n]");
				break;

			case 's':
				set_int_param(&(params->num_states), optarg, "--states [-s]");
				break;

			case 'V':
				version();
				break;

			case 'w':
				set_int_param(&(params->num_wave_vectors), optarg,
						"--wavevectors [-w]");
				break;

			case 'x':
				params->exact_solver = strndup(optarg, SOLVER_MAX_LEN);
				break;

			case '?':
				// error printed by getopt
				mpi_fail("Exiting due to unrecognized parameter.\n");
				break;

			default:
				mpi_fail("Exiting.\n");
				break;

		};

	}

	// Do further argument processing
	if (set_debug_iterative) {
		params->run_exact_solver = true;
		params->run_iterative_solver = true;
		params->keep_exact_solution = true;

		if (set_run_exact || set_run_iterative) {
			mpi_printf("(II) Debugging iterative solver, ignoring --exact and "
					"--iterative\n");
		}
	}
	else {
		if (set_run_exact && !set_run_iterative) {
			params->run_exact_solver = true;
			params->run_iterative_solver = false;
		}
		if (set_run_iterative && !set_run_exact) {
			params->run_exact_solver = false;
			params->run_iterative_solver = true;
		}
	}
}

int get_diag_mode(const char *diag_param)
{
	int mode = -1;
	const char *diag;

	if (diag_param == NULL) {
		diag = getenv("TOYCODE_DIAG");
	}
	else {
		diag = diag_param;
	}

	if(diag) {
		if(!strncmp(diag, "ZHEEV", 5)) {
			mode = 0;
		}
		if(!strncmp(diag, "ZHEEVD", 6)) {
			mode = 1;
		}
		if(!strncmp(diag, "ZHEEVR", 6)) {
			mode = 2;
		}
#ifdef DFTOY_USE_SCALAPACK
		if(!strncmp(diag, "PZHEEV", 6)) {
			mode = 3;
		}
		if(!strncmp(diag, "PZHEEVD", 7)) {
			mode = 4;
		}
		if(!strncmp(diag, "PZHEEVR", 7)) {
			mode = 5;
		}
#endif
#ifdef DFTOY_USE_ELPA
		if(!strncmp(diag, "ELPA", 4)) {
			mode = 6;
		}
#endif

		if(mode == -1) {
			mpi_printf("(II) Unknown diagonaliser %s, defaulting to ZHEEV.\n", diag);
			mode = 0;
		}
	}

	return mode;
}

// Print version information
void version()
{
	mpi_printf("DFToy version %s.\n", get_version_string());

	finalise_parallel();
	exit(EXIT_SUCCESS);
}

// Print usage information
void usage()
{
	mpi_printf(
			"DFToy version %s.\n\n"
			"Usage:\n"
			"mpirun -np <nprocs> dftoy <params>\n\n"
			"Default settings are equivalent to <dftoy -i -w 12 -s 4 >\n\n"
			"Optional parameters:\n\n"
			" -d, --debug_iterative\n"
			"            Debug iterative solver (forces -e -i).\n\n"
			" -e, --exact\n"
			"            Enable exact solver.\n\n"
			" -i, --iterative\n"
			"            Enable iterative solver.\n\n"
			" -b <n>, --bands=<n>\n"
			"            See '-s'\n"
			" -n <n>, --nl_states=<n>\n"
			"            Specify number of non-local states (size of beta).\n\n"
			" -s <n>, --states=<n>\n"
			"            Calculate <n> eigenstates.\n\n"
			" -h, --help\n"
			" -u, --usage\n"
			"            Display this usage info.\n\n"
			" -V, --version\n"
			"            Print version information.\n\n"
			" -w <n>, --wavevectors=<n>\n"
			"            Set wavevector resolution to <n>.\n\n"
			" -x <n>, --exact_solver=<solver>\n"
			"            Calculate exact solution using <solver>.\n\n"
			" -c <x.x>, --cutoff=<x.x>\n"
			"            Set the cutoff energy.\n\n"
			" -k <n>, --kpoints=<n>\n"
			"            Number of K-points in calculation (only effective for predictions)\n",
			get_version_string()
		);

	finalise_parallel();
	exit(EXIT_SUCCESS);
}

// Return version string. TODO: actually keep track of version...
// Major/minor? Date? Store in header/struct/etc else?
const char *get_version_string()
{
	return "0.1-unreleased";
}

void process_params(struct toycode_params *params) {
	params->cutoff_radius = sqrt(2.0*params->cutoff_energy);

	params->grid_xyz = ceil(params->grid_scale*params->cutoff_radius*10.26/CONST_PI);
	params->grid_dim = 2*params->grid_xyz+1;
	params->grid_size = pow(params->grid_dim, 3);

	//TODO this is nonsense, but need to keep the code running for now
	params->num_wave_vectors = ceil(params->cutoff_radius);
}

void debug_print_params(struct toycode_params *params) {
	mpi_printf("Cutoff Energy:\t%f\n", params->cutoff_energy);
	mpi_printf("Cutoff Radius:\t%f\n", params->cutoff_radius);
	mpi_printf("Grid scale:\t%f\n", params->grid_scale);
	mpi_printf("Grid XYZ:\t%d\n", params->grid_xyz);
	mpi_printf("Grid dim:\t%d\n", params->grid_dim);
	mpi_printf("Grid size:\t%d\n", params->grid_size);
	mpi_printf("num_states:\t%d\n", params->num_states);
	mpi_printf("num_nl_states:\t%d\n", params->num_nl_states);
}

