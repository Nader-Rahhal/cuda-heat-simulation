# GLFW config
GLFW_VERSION = 3.4
GLFW_TARBALL = glfw-$(GLFW_VERSION).zip
GLFW_DIR = glfw-$(GLFW_VERSION)
GLFW_BUILD_DIR = $(GLFW_DIR)/build
GLFW_LIB = $(GLFW_BUILD_DIR)/src/libglfw3.a

.PHONY: glfw clean

# Top-level GLFW build target
glfw:
	$(MAKE) $(GLFW_LIB)

# Download GLFW tarball
$(GLFW_TARBALL):
	wget https://github.com/glfw/glfw/releases/download/$(GLFW_VERSION)/$(GLFW_TARBALL)

# Unzip GLFW
$(GLFW_DIR): $(GLFW_TARBALL)
	unzip -o $(GLFW_TARBALL)

# Build GLFW from source
$(GLFW_LIB): $(GLFW_DIR)
	mkdir -p $(GLFW_BUILD_DIR) && \
	cd $(GLFW_BUILD_DIR) && \
	cmake .. -G "Unix Makefiles" && \
	make

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
TARGET_2DGL = 2Dgl

SRC_1D = heat_1d.cu
SRC_2D = heat_2d.cu
SRC_2DGL = heat_1d.cu

all: $(TARGET_1D) $(TARGET_2D)

1d: $(TARGET_1D)

2d: $(TARGET_2D)

2dgl: glfw $(TARGET_2DGL)

$(TARGET_1D): $(SRC_1D)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

$(TARGET_2D): $(SRC_2D)
	$(NVCC) $(NVCCFLAGS) -o $@ $<


$(TARGET_2DGL): $(SRC_2DGL)
	$(NVCC) $(NVCCFLAGS) -I$(GLFW_DIR)/include -c $< -o $@.o
	$(CXX) $@.o -L$(GLFW_BUILD_DIR)/src -lglfw3 \
		-lGL -lX11 -lpthread -ldl -lm \
		-L/usr/local/cuda/lib64 -lcudart -o $@

clean:
	rm -f $(TARGET_1D) $(TARGET_2D)
	rm -f $(GLFW_TARBALL)
