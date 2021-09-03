#ifndef TC_TRACE_H
#define TC_TRACE_H

#include <stdbool.h>
#include <sys/time.h>

struct tc_timer {
	struct timeval start;
	struct timeval stop;
	char *label;
	bool cleared;
};

void *TRACEMALLOC(size_t size);
void *TRACECALLOC(size_t nmemb, size_t size);
void *TRACEFFTW_MALLOC(size_t size);
void TRACEFREE(void *ptr);
void TRACEFFTW_FREE(void *ptr);
void RESET_MEMSTATS();
void report_mem_stats();

struct tc_timer create_timer(const char *label);
void start_timer(struct tc_timer *timer);
void stop_timer(struct tc_timer *timer);
void reset_timer(struct tc_timer *timer);
void destroy_timer(struct tc_timer *timer);
void report_timer(struct tc_timer *timer);

#endif
