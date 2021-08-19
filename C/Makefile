# Lines for GNU
OMPI_CC=gcc
CC=mpicc
CFLAGS=-std=c99 -g -O3
LAPACKLIBS=-llapacke
LIBS= $(LAPACKLIBS) -lfftw3 -lm
INCFLAGS=-I/usr/include/ -I./include
LDFLAGS=-L./ofiles

TARGET=.
SOLVERS_CPU=TRUE
export SOLVERS_CPU

NTASKS?=1
TOYCODE_DIAG?=PZHEEV

.PHONY: solvers

all: eigensolver 
	echo "libs:" $(LIBS)

#eigensolver: lib solvers
eigensolver: lib solvers
	$(CC) -o eigensolver ofiles/*.o  $(LIBS) $(LDFLAGS)


solvers:
	$(MAKE) --directory=src/solvers

lib:
	mkdir -p ofiles
	$(CC) $(CFLAGS) $(INCFLAGS) $(LDFLAGS) -o ofiles/main.o -c src/main.c
	$(CC) $(CFLAGS) $(INCFLAGS) $(LDFLAGS) -o ofiles/exact.o -c src/exact.c
	$(CC) $(CFLAGS) $(INCFLAGS) $(LDFLAGS) -o ofiles/iterative.o -c src/iterative.c
	$(CC) $(CFLAGS) $(INCFLAGS) $(LDFLAGS) -o ofiles/trace.o -c src/trace.c

clean:
	rm -rf ofiles
	rm eigensolver

run:
	TOYCODE_DIAG=$(TOYCODE_DIAG) mpirun -np $(NTASKS) --oversubscribe ./eigensolver

