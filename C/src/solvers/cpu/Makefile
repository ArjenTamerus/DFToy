# Lines for GNU
CC=mpicc
INCFLAGS=-I/home/at748/fftw/a/include/ -I/home/at748/elpa/include/elpa-2020.11.001 -I/usr/include/ -I../../../include
ELPA_CFLAGS=-I/usr/include/elpa-2020.11.001
#ELPA_CFLAGS=-lelpa_openmp -L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lmkl_blacs_openmpi_lp64 -lgomp -lpthread -lm -ldl -fopenmp -lgfortran

TARGET=../../..

all:
	$(CC) -g  ${INCFLAGS}  $(LDFLAGS) -c -fPIC -o $(TARGET)/diag.o diag.c
	$(CC) -g  ${INCFLAGS}  $(ELPA_CFLAGS) -c -fPIC -o $(TARGET)/elpa.o elpa.c

