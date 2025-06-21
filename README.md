# cuda-heat-simulation
CUDA Heat Diffusion Simulation is a high-performance 2D simulation of thermal diffusion across a grid, accelerated using NVIDIA CUDA. This project models how heat propagates through a surface over time using finite difference methods and parallel stencil computation.

# high level goals
- Expand to library functions that are passed a temperature buffer, alpha value, distance, and iterations
- Make playback and live xvisualizations possible with OpenGL
- Make it possible to run simulations that produce and don't produce visual output
- Add 3D support
- Enhance 1D and 2D support to be able to handle non-uniform distances between cells and varying border temperatures
- Implement testing
- Use CMake

