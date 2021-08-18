# Lines for GNU
CC=mpicc
INCFLAGS=-I/usr/include/ -I../../../include
#ELPA_CFLAGS=-I/usr/include/elpa-2020.11.001
#ELPA_CFLAGS=-lelpa_openmp -L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lmkl_blacs_openmpi_lp64 -lgomp -lpthread -lm -ldl -fopenmp -lgfortran

TARGET=../../..

all:
	$(CC) -g -O3 ${INCFLAGS} $(LDFLAGS) -c -o $(TARGET)/ofiles/diag.o diag.c
#	$(CC) -g  ${INCFLAGS}  $(ELPA_CFLAGS) -c -fPIC -o $(TARGET)/elpa.o elpa.c

