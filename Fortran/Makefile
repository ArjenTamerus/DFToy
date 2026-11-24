all:
	gfortran -g -c parameters.f90
	gfortran -g -I/usr/include -c fft.f90
	gfortran -g -c hamiltonian.f90
	gfortran -g -c dftoy.f90
	gfortran -g -o dftoy dftoy.o hamiltonian.o parameters.o -llapack -lblas -lfftw3

clean:
	rm *.mod *.o dftoy

run:
	./dftoy
