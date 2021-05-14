# Lines for GNU
OMPI_CC=gcc
CC=mpicc -g
#LAPACKLIBS=-llapack -lblas
LAPACKLIBS= -lelpa -lscalapack -lopenblas -lgfortran
#LAPACKLIBS=`pkg-config --libs mkl-dynamic-lp64-seq`
#LAPACKLIBS=-DMKL_ILP64 -I/opt/intel/mkl/include -L/opt/intel/mkl/lib/intel64 -lmkl_scalapack_lp64 -Wl,--no-as-needed -lmkl_gf_lp64 -lmkl_gnu_thread -lmkl_core -lmkl_blacs_intelmpi_lp64 -lgomp -lpthread -lm -ldl
LIBS= $(LAPACKLIBS) -lfftw3 -lm
INCFLAGS= -I/home/at748/elpa/include/elpa-2020.11.001 -I/home/at748/fftw/a/include/ -I/usr/include/ -I./include
LDFLAGS= -L/home/at748/elpa/lib -L/home/at748/fftw/a/lib -L. -Wl,-rpath=. -Bdynamic $(LIBS)
#LDFLAGS=-L/home/at748/fftw/a/lib -L. -Wl,-rpath=. -Bdynamic -DMKL_LP64 -I/opt/intel/mkl/include -L/opt/intel/mkl/lib/intel64 -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl -lfftw3
TARGET=.
SOLVERS_CPU=TRUE
export SOLVERS_CPU

NTASKS?=1
TOYCODE_DIAG?=PZHEEV

.PHONY: solvers

all: eigensolver 
	echo "libs:" $(LIBS)

eigensolver: lib solvers
	$(CC) -g -O3 ${INCFLAGS} -c src/eigensolver.c  $(LIBS) $(LDFLAGS)
	$(CC) -g -O3  -o eigensolver *.o  $(LIBS) $(LDFLAGS)


solvers:
	$(MAKE) --directory=src/solvers
	#$(CC) -g  ${INCFLAGS}  $(LDFLAGS) -c -fPIC src/solvers/cpu/diag.c

lib:
	$(CC) -g  ${INCFLAGS}  $(LDFLAGS) -c -fPIC src/c_interface.c
	$(CC) -g  ${INCFLAGS}  $(LDFLAGS) -c -fPIC src/trace.c

clean:
	rm -f *.a *.o *.so
	rm eigensolver

run:
	TOYCODE_DIAG=$(TOYCODE_DIAG) mpirun -np $(NTASKS) --oversubscribe ./eigensolver

