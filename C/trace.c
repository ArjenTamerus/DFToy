#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>

#include <fftw3.h>
#include "trace.h"

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
	printf("Total memory used: %d MB.\n", max_mem_use/(1024*1024));
	printf("Current memory use: %d MB.\n", current_mem_use/(1024*1024));
}

int mpi_printf(int comm_rank, const char *format, ...)
{
	int status = 0;
	va_list myargs;

	if(0 == comm_rank) {
		va_start(myargs, format);
		status = vprintf(format, myargs);
		va_end(myargs);
	}

	return status;
}

