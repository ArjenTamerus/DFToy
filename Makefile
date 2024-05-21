# Setup for Intel oneAPI Toolkit + fftw
export I_MPI_CC=icx
export I_MPI_CXX=icpx
CC=mpicc
CXX=mpicxx
CPP=cpp
CFLAGS=-xHost -g -std=c99 -O3 -D_POSIX_C_SOURCE=200809L -qmkl=parallel -fiopenmp
CXXFLAGS=-xHost -std=c++20 -fsycl -g -O3 -qmkl=parallel -fiopenmp

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
#	SCALAPACK_LIBS=-lscalapack-openmpi
	SCALAPACK_LDFLAGS=-lmkl_scalapack_lp64 -lmkl_blacs_intelmpi_lp64
endif


CPPFLAGS=-I$(MKLROOT)/include/fftw -I/usr/include/ -I./include $(SCALAPACK_CPPFLAGS) $(LAPACK_CPPFLAGS) \
				 $(ELPA_CPPFLAGS)
LDFLAGS=$(SCALAPACK_LDFLAGS) $(LAPACK_LDFLAGS) $(ELPA_LDFLAGS) -fiopenmp
LDLIBS=$(SCALAPACK_LIBS) $(LAPACK_LIBS) $(ELPA_LIBS) -L${MKLROOT}/lib  -lmkl_sycl_blas -lmkl_sycl_lapack -lmkl_sycl_dft -lsycl -lfftw3x_cdft_lp64  -lmkl_cdft_core -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -lmkl_blacs_intelmpi_lp64 -liomp5 -lpthread -lm -ldl

TARGET=dftoy

SRCDIR=src
OBJDIR=obj
SRC=$(wildcard $(SRCDIR)/*.c)
OBJ=$(SRC:$(SRCDIR)/%.c=$(OBJDIR)/%.o)
CPPSRCDIR=src/sycl
CPPOBJDIR=obj
CPPSRC=$(wildcard $(CPPSRCDIR)/*.cpp)
CPPOBJ=$(CPPSRC:$(CPPSRCDIR)/%.cpp=$(CPPOBJDIR)/%.o)

export CFLAGS
export CPPFLAGS

NTASKS?=1
TOYCODE_DIAG?=ZHEEVD

.PHONY: all clean run

all: $(TARGET)

$(TARGET): solvers $(OBJ) $(CPPOBJ)
	$(CXX) -fsycl -o $@ $(LDFLAGS) $(OBJDIR)/*.o $(LDLIBS)

$(OBJDIR)/%.o: $(SRCDIR)/%.c | $(OBJDIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

$(CPPOBJDIR)/%.o: $(CPPSRCDIR)/%.cpp | $(CPPOBJDIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(OBJDIR):
	mkdir -p $@

solvers:
	$(MAKE) --directory=src/solvers

clean:
	rm -rf $(OBJDIR)
	rm $(TARGET)

run:
	 mpirun -np $(NTASKS) --oversubscribe ./$(TARGET) -x $(TOYCODE_DIAG)

