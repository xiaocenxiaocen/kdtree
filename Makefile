CXX = g++ -O3 -Wall -g -std=c++0x -DDEBUG
CFLAGS = -fopenmp -fpic
LDFLAGS = -lm -lpthread

target = libkdtree.so \
test

all: $(target)

libkdtree.so: kdtree.o
	$(CXX) -fopenmp -shared -o $@ $< -lm -lpthread

test: test.o
	$(CXX) -fopenmp -o $@ $< -L. -lm -lpthread -lkdtree

.SUFFIXES: .cpp .o

.cpp.o:
	$(CXX) $(CFLAGS) -c $< -o $@

.PHONY: clean

clean:
	-rm *.o
	-rm *.so
	-rm test
