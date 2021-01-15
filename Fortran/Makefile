# Lines for GNU
CC=gcc
F90=gfortran
ELEC=-lhamiltonian
INCFLAGS=-I/usr/include/ -I.
LDFLAGS=-L. -Wl,-rpath=. -Bdynamic -llapack -lblas -lfftw3
TARGET=.

all: eigensolver eigensolver_sol6_omp

eigensolver: lib
	$(F90) -o eigensolver eigensolver.f90  $(ELEC)  -lm $(LDFLAGS)

eigensolver_sol6_omp: lib
	$(F90) -g -pg -o eigensolver_solution6_omp eigensolver_solution6_omp.f90  $(ELEC)  -lm $(LDFLAGS)

lib: hamiltonian.f90
	$(F90) -O3 ${INCFLAGS}  $(LDFLAGS) -c -fPIC hamiltonian.f90 -lfftw3
	$(CC)  ${INCFLAGS}  $(LDFLAGS) -c -fPIC c_interface.c
	ar r libhamiltonian.a c_interface.o hamiltonian.o
	ranlib libhamiltonian.a
	$(F90)   $(LDFLAGS) -shared -Wl,-soname,libhamiltonian.so -o libhamiltonian.so c_interface.o  hamiltonian.o -lfftw3

clean:
	rm -f *.a *.o *.mod *.so
	rm eigensolver

