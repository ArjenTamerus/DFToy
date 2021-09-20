# Lines for GNU
OMPI_CC=gcc
CC=mpicc
CPPFLAGS+=-I../../include

OBJDIR=../../obj
SRCDIR=.

SRC=$(wildcard $(SRCDIR)/*.c)
OBJ=$(SRC:$(SRCDIR)/%.c=$(OBJDIR)/%.o)

all: $(OBJ)

$(OBJDIR)/%.o: $(SRCDIR)/%.c | $(OBJDIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

$(OBJDIR):
	mkdir -p $@

