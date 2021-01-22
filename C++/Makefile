# Lines for GNU
CC=gcc
CXX=g++
CXXFLAGS= -Wall -Wextra -std=c++17 -O3
F90=gfortran
ELEC=-lhamiltonian
INCFLAGS=-I/usr/include/ -I.
LDFLAGS=-L. -Wl,-rpath=. -Bdynamic -llapack -lblas -lfftw3
TARGET=.

all: eigensolver

eigensolver: lib
	$(CXX) ${INCFLAGS} ${CXXFLAGS} -c eigensolver.cpp  $(ELEC)  -lm $(LDFLAGS)
	$(F90) -o eigensolver eigensolver.o  $(ELEC)  -lm $(LDFLAGS)   -lstdc++

lib: hamiltonian.f90
	$(F90) ${INCFLAGS}  $(LDFLAGS) -c -fPIC hamiltonian.f90 -lfftw3
	$(CC)  ${INCFLAGS}  $(LDFLAGS) -c -fPIC hamiltonian_c.cpp
	ar r libhamiltonian.a hamiltonian_c.o hamiltonian.o
	ranlib libhamiltonian.a
	$(F90)   $(LDFLAGS) -shared -Wl,-soname,libhamiltonian.so -o libhamiltonian.so hamiltonian_c.o  hamiltonian.o -lfftw3

clean:
	rm -f *.a *.o *.mod *.so
	rm -f eigensolver

