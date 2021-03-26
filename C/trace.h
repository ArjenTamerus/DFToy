void *TRACEMALLOC(size_t size);
void *TRACECALLOC(size_t nmemb, size_t size);
void *TRACEFFTW_MALLOC(size_t size);
void TRACEFREE(void *ptr);
void TRACEFFTW_FREE(void *ptr);
void RESET_MEMSTATS();
void report_mem_stats();
