CC ?= gcc
CFLAGS ?= -std=c11 -O2 -Wall -Wextra -Iinclude
LDFLAGS ?= -lm

BUILD := build
LIB := $(BUILD)/libfp4_ref.a
TEST := $(BUILD)/test_fp4_ref

.PHONY: all test clean

all: $(LIB) $(TEST)

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/fp4_ref.o: src/fp4_ref.c include/fp4_ref.h | $(BUILD)
	$(CC) $(CFLAGS) -c $< -o $@

$(LIB): $(BUILD)/fp4_ref.o
	ar rcs $@ $^

$(TEST): tests/test_fp4_ref.c $(LIB)
	$(CC) $(CFLAGS) $< $(LIB) $(LDFLAGS) -o $@

test: $(TEST)
	./$(TEST)

clean:
	rm -rf $(BUILD)
