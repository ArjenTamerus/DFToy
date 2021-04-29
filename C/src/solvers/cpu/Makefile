# Lines for GNU
CC=mpicc
INCFLAGS=-I/home/at748/fftw/a/include/ -I/usr/include/ -I../../../include
ELPA_CFLAGS=-I/usr/include/elpa_openmp-2020.11.001 -fopenmp
TARGET=../../..

all:
	$(CC) -g  ${INCFLAGS}  $(LDFLAGS) -c -fPIC -o $(TARGET)/diag.o diag.c
	$(CC) -g  ${INCFLAGS}  $(ELPA_CFLAGS) -c -fPIC -o $(TARGET)/elpa.o elpa.c

