# Makefile for hw2_sift project
# Compiler and flags
MPICXX = mpicxx
MPIFLAGS = -std=c++17 -O3 -fopenmp
INCLUDES = -Iinclude

# Source files
SOURCES = src/hw2.cpp src/sift.cpp src/image.cpp
OBJECTS = src/hw2.o src/sift.o src/image.o
TARGET = hw2

# STB image library files
STB_HEADERS = include/stb_image.h include/stb_image_write.h

# Default target
all: $(TARGET)

# Build targets
$(TARGET): $(OBJECTS)
	$(MPICXX) $(MPIFLAGS) -o $(TARGET) $(OBJECTS)

# Compile source files to object files
src/hw2.o: src/hw2.cpp include/sift.hpp include/image.hpp
	$(MPICXX) $(MPIFLAGS) $(INCLUDES) -c src/hw2.cpp -o src/hw2.o

src/sift.o: src/sift.cpp include/sift.hpp include/image.hpp $(STB_HEADERS)
	$(MPICXX) $(MPIFLAGS) $(INCLUDES) -c src/sift.cpp -o src/sift.o

src/image.o: src/image.cpp include/image.hpp $(STB_HEADERS)
	$(MPICXX) $(MPIFLAGS) $(INCLUDES) -c src/image.cpp -o src/image.o

# Clean up build artifacts
clean:
	rm -f $(OBJECTS) $(MPI_OBJECTS) $(TARGET) result.jpg

.PHONY: all clean
