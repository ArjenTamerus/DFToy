# Lines for GNU
OMPI_CC=gcc
CC=mpicc -g
#LAPACKLIBS=-llapack -lblas
LAPACKLIBS= -lscalapack -lopenblas -lgfortran
#LAPACKLIBS=`pkg-config --libs mkl-dynamic-lp64-seq`
#LAPACKLIBS=-DMKL_ILP64 -I/opt/intel/mkl/include -L/opt/intel/mkl/lib/intel64 -lmkl_scalapack_lp64 -Wl,--no-as-needed -lmkl_gf_lp64 -lmkl_gnu_thread -lmkl_core -lmkl_blacs_intelmpi_lp64 -lgomp -lpthread -lm -ldl
LIBS=-lhamiltonian $(LAPACKLIBS) -lfftw3 -lm
INCFLAGS=-I/home/at748/fftw/a/include/ -I/usr/include/ -I.
LDFLAGS= -L/home/at748/fftw/a/lib -L. -Wl,-rpath=. -Bdynamic $(LIBS)
#LDFLAGS=-L/home/at748/fftw/a/lib -L. -Wl,-rpath=. -Bdynamic -DMKL_LP64 -I/opt/intel/mkl/include -L/opt/intel/mkl/lib/intel64 -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl -lfftw3
TARGET=.

NTASKS?=1
TOYCODE_DIAG?=PZHEEV

all: eigensolver 
	echo "libs:" $(LIBS)

eigensolver: lib
	$(CC) -g -O3 ${INCFLAGS} -c eigensolver.c  $(LIBS) $(LDFLAGS)
	$(CC) -g -O3  -o eigensolver eigensolver.o  $(LIBS) $(LDFLAGS)

lib:
	$(CC) -g  ${INCFLAGS}  $(LDFLAGS) -c -fPIC c_interface.c
	$(CC) -g  ${INCFLAGS}  $(LDFLAGS) -c -fPIC trace.c
	$(CC) -g  ${INCFLAGS}  $(LDFLAGS) -c -fPIC diag.c
	ar r libhamiltonian.a c_interface.o trace.o diag.o
	ranlib libhamiltonian.a
	$(CC) -g   $(LDFLAGS) -shared -Wl,-soname,libhamiltonian.so -o libhamiltonian.so c_interface.o trace.o diag.o

clean:
	rm -f *.a *.o *.mod *.so
	rm eigensolver

run:
	TOYCODE_DIAG=$(TOYCODE_DIAG) mpirun -np $(NTASKS) --oversubscribe ./eigensolver

