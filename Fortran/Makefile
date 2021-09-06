# Lines for GNU
CC=gcc
F90=gfortran
ELEC=-lhamiltonian
INCFLAGS=-I/usr/include/ -I.
LDFLAGS=-L. -Wl,-rpath=. -Bdynamic -llapack -lblas -lfftw3
TARGET=.

all: dftoy

dftoy: lib
	$(F90) -o dftoy dftoy.f90  $(ELEC)  -lm $(LDFLAGS)

lib: hamiltonian.f90
	$(F90) -O3 ${INCFLAGS}  $(LDFLAGS) -c -fPIC hamiltonian.f90 -lfftw3
	#$(CC)  ${INCFLAGS}  $(LDFLAGS) -c -fPIC c_interface.c
	#ar r libhamiltonian.a hamiltonian.o
	#ranlib libhamiltonian.a
	$(F90)   $(LDFLAGS) -shared -Wl,-soname,libhamiltonian.so -o libhamiltonian.so  hamiltonian.o -lfftw3

clean:
	rm -f *.a *.o *.mod *.so
	rm dftoy

