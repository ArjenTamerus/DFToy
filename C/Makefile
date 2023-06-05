# Setup for Intel oneAPI Toolkit + fftw
export I_MPI_CC=icx
CC=mpicc
CPP=cpp
CFLAGS=-xHost -g -std=c99 -O3 -D_POSIX_C_SOURCE=200809L -qmkl=parallel -fiopenmp

LAPACK_LIBS=-mkl
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


CPPFLAGS=-I$(FFTW_ROOT)/include -I/usr/include/ -I./include $(SCALAPACK_CPPFLAGS) $(LAPACK_CPPFLAGS) \
				 $(ELPA_CPPFLAGS)
LDFLAGS=-L$(FFTW_ROOT)/lib $(SCALAPACK_LDFLAGS) $(LAPACK_LDFLAGS) $(ELPA_LDFLAGS) -fiopenmp
LDLIBS=$(SCALAPACK_LIBS) $(LAPACK_LIBS) $(ELPA_LIBS) -lfftw3_mpi -lfftw3 -qmkl=parallel -lm -L${MKLROOT}/lib/intel64 -lpthread -lm -ldl

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

