# Lines for GNU
CC=mpicc
INCFLAGS=-I/usr/include/ -I../../../include
#ELPA_CFLAGS=-I/usr/include/elpa-2020.11.001

TARGET=../../..

all:
	$(CC) -g -O3 ${INCFLAGS} $(LDFLAGS) -c -o $(TARGET)/ofiles/diag.o diag.c
#	$(CC) -g  ${INCFLAGS}  $(ELPA_CFLAGS) -c -fPIC -o $(TARGET)/elpa.o elpa.c

