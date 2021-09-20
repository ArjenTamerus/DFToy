# Lines for GNU
OMPI_CC=gcc
CC=mpicc
CPP=cpp
CFLAGS=-std=c99 -g -O3 -Wall -Wpedantic -D_POSIX_C_SOURCE=200809L

LAPACK_LIBS=-llapacke
LAPACK_LDFLAGS=

ifdef WITH_ELPA
	export WITH_SCALAPACK=true
	ELPA_CPPFLAGS=-isystem/usr/include/elpa -DDFTOY_USE_ELPA
	ELPA_LIBS=-lelpa
	ELPA_LDFLAGS=-L/usr/lib/x86_64-linux-gnu
endif

ifdef WITH_SCALAPACK
	SCALAPACK_CPPFLAGS=-DDFTOY_USE_SCALAPACK
	SCALAPACK_LIBS=-lscalapack-openmpi
	SCALAPACK_LDFLAGS=
endif


CPPFLAGS=-I/usr/include/ -I./include $(SCALAPACK_CPPFLAGS) $(LAPACK_CPPFLAGS) \
				 $(ELPA_CPPFLAGS)
LDFLAGS=$(SCALAPACK_LDFLAGS) $(LAPACK_LDFLAGS) $(ELPA_LDFLAGS)
LDLIBS=$(SCALAPACK_LIBS) $(LAPACK_LIBS) $(ELPA_LIBS) -lfftw3 -lm

TARGET=dftoy

SRCDIR=src
OBJDIR=obj
SRC=$(wildcard $(SRCDIR)/*.c)
OBJ=$(SRC:$(SRCDIR)/%.c=$(OBJDIR)/%.o)

export CFLAGS
export CPPFLAGS

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

