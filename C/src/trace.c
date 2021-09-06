/*
 * Trace.h
 *
 * Memory tracking and timing routines. Pretty much a stub right now.
 *
 */

#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <string.h>

#include <fftw3.h>

#include <sys/time.h>

#include "trace.h"
#include "parallel.h"

size_t max_mem_use = 0;
size_t current_mem_use = 0;

void *TRACEMALLOC(size_t size)
{
	current_mem_use += size;
	max_mem_use += size;
	return malloc(size);
}

void *TRACECALLOC(size_t nmemb, size_t size)
{
	current_mem_use += nmemb*size;
	max_mem_use += nmemb*size;
	return calloc(nmemb, size);
}

void *TRACEFFTW_MALLOC(size_t size)
{
	current_mem_use += size;
	max_mem_use += size;
	return fftw_malloc(size);
}

void TRACEFREE(void *ptr)
{
	free(ptr);
}

void TRACEFFTW_FREE(void *ptr)
{
	fftw_free(ptr);
}

void RESET_MEMSTATS()
{
	current_mem_use = 0;
	max_mem_use = 0;
}

void report_mem_stats()
{
	mpi_printf("Memory: %d\n", max_mem_use/(1024*1024));
}

struct tc_timer create_timer(const char *label)
{
	if (label == NULL) {
		mpi_fail("Invalid label in timer_start!\n");
	}

	struct tc_timer timer;
	timer.label = strdup(label);
	timer.cleared = true;

	return timer;
}

void start_timer(struct tc_timer *timer)
{
	if (timer == NULL) {
		mpi_error("Invalid timer in timer_start!\n");
	}

	if (!timer->cleared) {
		mpi_error("Timer \"%s\" is already running!\n", timer->label);
	}

	timer->cleared = false;
	gettimeofday(&(timer->start), NULL);
}

void stop_timer(struct tc_timer *timer)
{
	if (timer == NULL) {
		mpi_error("Invalid timer!\n");
	}

	if (timer->cleared) {
		mpi_error("Timer \"%s\" is not running!\n", timer->label);
	}

	gettimeofday(&(timer->stop), NULL);
}

void reset_timer(struct tc_timer *timer)
{
	if (timer == NULL) {
		mpi_error("Invalid timer in timer_reset!\n");
	}

	timer->cleared = true;
}

void destroy_timer(struct tc_timer *timer)
{
	if (timer == NULL) {
		mpi_error("Invalid timer in timer_destroy!\n");
	}

	if (timer->label != NULL) {
		free(timer->label);
	}

	timer->cleared = true;
}

void report_timer(struct tc_timer *timer)
{
	mpi_printf("Timed region \"%s\" took %f seconds.\n",
			timer->label,
			(double)(timer->stop.tv_sec - timer->start.tv_sec) +
			(double)(timer->stop.tv_usec - timer->start.tv_usec)/1000000.0)
			;
}
