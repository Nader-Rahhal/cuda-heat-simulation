# Compilers
CC = gcc-9
CXX = g++-9
NVCC = nvcc

# Flags
CFLAGS = -Wall -g
NVCCFLAGS = -ccbin=$(CXX) -O3 --std=c++14

# Targets
TARGET_CU = maincu
SRC_CU = heat_1d.cu

all: $(TARGET_CU)

$(TARGET_CU): $(SRC_CU)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -f $(TARGET_C) $(TARGET_CU)

