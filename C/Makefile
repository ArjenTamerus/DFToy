# Lines for GNU
CC=gcc
F90=gfortran
ELEC=-lhamiltonian
INCFLAGS=-I/usr/include/ -I.
LDFLAGS=-L. -Wl,-rpath=. -Bdynamic -llapack -lblas -lfftw3
TARGET=.

#all: eigensolver eigensolver_complete eigensolver_sol2_sd eigensolver_sol3_p_sd eigensolver_sol3_cg eigensolver_sol4_p_cg
all: eigensolver eigensolver_sol3_p_sd eigensolver_sol3_cg eigensolver_sol4_p_cg eigensolver_complete

eigensolver: lib
	$(CC) -O3 ${INCFLAGS} -c eigensolver.c  $(ELEC)  -lm $(LDFLAGS)
	$(F90) -O3  -o eigensolver eigensolver.o  $(ELEC)  -lm $(LDFLAGS)

eigensolver_sol1: lib
	$(CC) ${INCFLAGS} -c eigensolver_solution1.c  $(ELEC)  -lm $(LDFLAGS)
	$(F90) -o eigensolver_sol1 eigensolver_solution1.o  $(ELEC)  -lm $(LDFLAGS)

eigensolver_sol2_sd: lib
	$(CC) ${INCFLAGS} -pg -c eigensolver_solution2_steepest_descent.c  $(ELEC)  -lm $(LDFLAGS)
	$(F90) -pg -o eigensolver_sol2_sd eigensolver_solution2_steepest_descent.o  $(ELEC)  -lm $(LDFLAGS)

eigensolver_sol3_p_sd: lib
	$(CC) ${INCFLAGS} -c eigensolver_solution3_precondition_steepest_descent.c  $(ELEC)  -lm $(LDFLAGS)
	$(F90) -o eigensolver_sol3_p_sd eigensolver_solution3_precondition_steepest_descent.o  $(ELEC)  -lm $(LDFLAGS)

eigensolver_sol3_cg: lib
	$(CC) ${INCFLAGS} -c eigensolver_solution3_conjugate_gradient.c  $(ELEC)  -lm $(LDFLAGS)
	$(F90) -o eigensolver_sol3_cg eigensolver_solution3_conjugate_gradient.o  $(ELEC)  -lm $(LDFLAGS)

eigensolver_sol4_p_cg: lib
	$(CC) ${INCFLAGS} -c eigensolver_solution4_precondition_conjugate_gradient.c  $(ELEC)  -lm $(LDFLAGS)
	$(F90) -o eigensolver_sol4_p_cg eigensolver_solution4_precondition_conjugate_gradient.o  $(ELEC)  -lm $(LDFLAGS)

eigensolver_complete: lib
	$(CC) ${INCFLAGS} -g -c eigensolver_complete.c  $(ELEC)  -lm $(LDFLAGS)
	$(F90) -g -o eigensolver_complete eigensolver_complete.o  $(ELEC)  -lm $(LDFLAGS)

lib: hamiltonian.f90
	$(F90) ${INCFLAGS}  $(LDFLAGS) -c -fPIC hamiltonian.f90 -lfftw3
	$(CC)  ${INCFLAGS}  $(LDFLAGS) -c -fPIC c_interface.c
	ar r libhamiltonian.a c_interface.o hamiltonian.o
	ranlib libhamiltonian.a
	$(F90)   $(LDFLAGS) -shared -Wl,-soname,libhamiltonian.so -o libhamiltonian.so c_interface.o  hamiltonian.o -lfftw3

clean:
	rm -f *.a *.o *.mod *.so
	rm eigensolver

