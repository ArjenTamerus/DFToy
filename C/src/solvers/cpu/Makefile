# Lines for GNU
OMPI_CC=gcc
CC=mpicc
CFLAGS=-g -O3
CPPFLAGS=-I/usr/include/elpa -I../../../include
#ELPA_CFLAGS=-I/usr/include/elpa-2020.11.001

OBJDIR=../../../obj
SRCDIR=.

SRC=$(wildcard $(SRCDIR)/*.c)
OBJ=$(SRC:$(SRCDIR)/%.c=$(OBJDIR)/%.o)

all: $(OBJ)

$(OBJDIR)/%.o: $(SRCDIR)/%.c | $(OBJDIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

$(OBJDIR):
	mkdir -p $@

#all:
#	$(CC) -g -O3 ${INCFLAGS} -c -o $(TARGET)/ofiles/diag.o diag.c
#	$(CC) -g -O3 ${INCFLAGS} -c -o $(TARGET)/ofiles/elpa.o elpa.c

