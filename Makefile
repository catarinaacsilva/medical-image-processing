CC = g++
CFLAGS = -g -Wall

TARGET := main

SRCS := $(wildcard *.cpp)
OBJS := $(patsubst %.cpp,%.o,$(SRCS))

OPENCV = `pkg-config opencv4 --cflags --libs`
LIBS = $(OPENCV)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LIBS)

%.o: %.cpp
	$(CC) $(CFLAGS) -c $< $(LIBS)

clean:
	rm -rf $(TARGET) *.o

.PHONY: all clean