CC = g++
CFLAGS = -g -Wall -O2 -std=c++17 -pipe -march=native

SRCS := $(wildcard *.cpp)
OBJS := $(patsubst %.cpp,%.o,$(SRCS))

OPENCV = `pkg-config opencv4 --cflags --libs`
LIBS = $(OPENCV)

.PHONY: all clean

all: main train watershed

watershed: watershed.o
	$(CC) $(CFLAGS) -o $@ $^ $(LIBS)

main: main.o lib_od.o lib_oc.o lib_fs.o
	$(CC) $(CFLAGS) -o $@ $^ $(LIBS)

train: train.o lib_od.o lib_oc.o lib_fs.o
	$(CC) $(CFLAGS) -o $@ $^ $(LIBS)

%.o: %.cpp
	$(CC) $(CFLAGS) -c $< $(LIBS)

docs: %.cpp %.h %.hpp Doxyfile
	doxygen Doxyfile

clean:
	rm -rf main train *.o documentation