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
#include "parallel.h"
#include "interfaces.h"

struct option tc_params[] = {
	{"wavevectors", required_argument, 0, 'w'},
	{"states", required_argument, 0, 's'},
	{"exact_solver", required_argument, 0, 'x'},
	{"debug_iterative", no_argument, 0, 'd'},
	{"exact", no_argument, 0, 'e'},
	{"iterative", no_argument, 0, 'i'},
	{0, 0, 0, 0}
};

void set_default_configuration_params(struct toycode_params *params)
{
	params->num_wave_vectors = 3;
	params->num_states = 1;

	params->run_exact_solver = true;
	params->run_iterative_solver = true;
	params->keep_exact_solution = false;

	params->exact_solver = NULL;
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
		opt_id = getopt_long_only(argc, argv, "deis:w:x:", tc_params, NULL);

		if(opt_id == -1) {
			break;
		}
		switch(opt_id) {
			case 'd':
				set_debug_iterative = true;
				break;

			case 'e':
				set_run_exact = true;
				break;

			case 'i':
				set_run_iterative = true;
				break;

			case 's':
				set_int_param(&(params->num_states), optarg, "--states [-s]");
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
		if(!strncmp(diag, "PZHEEV", 6)) {
			mode = 3;
		}
		if(!strncmp(diag, "PZHEEVD", 7)) {
			mode = 4;
		}
		if(!strncmp(diag, "PZHEEVR", 7)) {
			mode = 5;
		}
		if(!strncmp(diag, "ELPA", 4)) {
			mode = 6;
		}

		if(mode == -1) {
			mpi_printf("(II) Unknown diagonaliser %s, defaulting to ZHEEV.\n", diag);
			mode = 0;
		}
	}

	return mode;
}

