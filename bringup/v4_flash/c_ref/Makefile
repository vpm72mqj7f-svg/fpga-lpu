# FP8 E4M3 reference implementation — build & test
CC      = gcc
CFLAGS  = -O2 -Wall -Wextra -std=c11
LDLIBS  = -lm

.PHONY: all run clean

all: test_fp8

fp8_e4m3.o: fp8_e4m3.c fp8_e4m3.h
	$(CC) $(CFLAGS) -c -o $@ fp8_e4m3.c

test_fp8.o: test_fp8.c fp8_e4m3.h
	$(CC) $(CFLAGS) -c -o $@ test_fp8.c

test_fp8: fp8_e4m3.o test_fp8.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

run: test_fp8
	./test_fp8

clean:
	rm -f *.o test_fp8
