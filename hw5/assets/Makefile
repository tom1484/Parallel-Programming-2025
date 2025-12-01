CXX := g++
HIPFLAGS := $(CXXFLAGS) --offload-arch=gfx908
TARGET := hw5


.PHONY: all
all: $(TARGET)

.PHONY: hw5
hw5: hw5.cpp
	hipcc $(HIPFLAGS) -o hw5 hw5.cpp

.PHONY: clean
clean:
	rm -f $(TARGET)


