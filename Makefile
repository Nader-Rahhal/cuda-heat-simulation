# Compilers
CC = gcc-9
CXX = g++-9
NVCC = nvcc

# Flags
CFLAGS = -Wall -g
NVCCFLAGS = -ccbin=$(CXX) -O3 --std=c++14

# Targets
TARGET_1D = 1D
TARGET_2D = 2D

SRC_1D = heat_1d.cu
SRC_2D = heat_2d.cu

all: $(TARGET_1D) $(TARGET_2D)

1d: $(TARGET_1D)

2d: $(TARGET_2D)

$(TARGET_1D): $(SRC_1D)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

$(TARGET_2D): $(SRC_2D)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -f $(TARGET_1D) $(TARGET_2D)

