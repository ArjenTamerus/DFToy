# Lines for GNU
OMPI_CC=gcc
CC=mpicc
CFLAGS=-std=c99 -g -O3 -Wall -Wpedantic -D_POSIX_C_SOURCE=200809L
LAPACK_LIBS=-lscalapack-openmpi -llapacke
ELPA_LIBS=-lelpa
LDLIBS= $(LAPACK_LIBS) $(ELPA_LIBS) -lfftw3 -lm
CPP=cpp

LAPACK_CPPFLAGS=
ELPA_CPPFLAGS=-I/usr/include/elpa
CPPFLAGS=-I/usr/include/ -I./include $(LAPACK_CPPFLAGS) $(ELPA_CPPFLAGS)

LAPACK_LDFLAGS=
ELPA_LDFLAGS=-L/usr/lib/x86_64-linux-gnu
LDFLAGS=$(LAPACK_LFLAGS) $(ELPA_LFLAGS) $(LIBS)

TARGET=dftoy

SRCDIR=src
OBJDIR=obj
SRC=$(wildcard $(SRCDIR)/*.c)
OBJ=$(SRC:$(SRCDIR)/%.c=$(OBJDIR)/%.o)

SOLVERS_CPU=TRUE
export SOLVERS_CPU

NTASKS?=1
TOYCODE_DIAG?=ZHEEVD

.PHONY: all clean run

all: $(TARGET)

$(TARGET): solvers $(OBJ)
	$(CC) -o $@ $(LDFLAGS) $(OBJDIR)/*.o $(LDLIBS)

$(OBJDIR)/%.o: $(SRCDIR)/%.c | $(OBJDIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

$(OBJDIR):
	mkdir -p $@

solvers:
	$(MAKE) --directory=src/solvers

clean:
	rm -rf $(OBJDIR)
	rm $(TARGET)

run:
	 mpirun -np $(NTASKS) --oversubscribe ./$(TARGET) -x $(TOYCODE_DIAG)

