#include <stdio.h>
#include <stdlib.h>
#include <fstream>

#define THERMAL_DIFFUSIVITY 100.0f
#define DISTANCE_BETWEEN_CELLS 100.0f

// want to iteraively add data to file after every iteration - can be normal C func called after kernel 

void write_to_output_1d(const char* filename, float* temps, int N) {
    FILE* fd = fopen(filename, "a"); // open in append mode
    if (!fd) {
        perror("Failed to open output file");
        return;
    }

    for (int i = 0; i < N; ++i) {
        fprintf(fd, "%.6f ", temps[i]);  // write each value with space
    }
    fprintf(fd, "\n");  // end of one iteration
    fclose(fd);
}



__global__ void heat_update(float *old_temps, float *new_temps, int N, float alpha, float dx) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= N) return;

    float curr_temp = old_temps[i];
    float laplacian = 0.0f;

    if (i == 0) {
        laplacian = old_temps[i + 1] - 2 * curr_temp;
    }
    else if (i == N - 1) {
        laplacian = old_temps[i - 1] - 2 * curr_temp;
    }
    else {
        laplacian = old_temps[i + 1] - 2 * curr_temp + old_temps[i - 1];
    }

    new_temps[i] = curr_temp + alpha * (laplacian / (dx * dx));
}

int main(void) {
    const int N = 4;
    float old_temps_host[] = {1.0, 2.0, 3.0, 4.0};

    float *old_temps_d, *new_temps_d;
    cudaMalloc((void **)&old_temps_d, N * sizeof(float));
    cudaMalloc((void **)&new_temps_d, N * sizeof(float));

    cudaMemcpy(old_temps_d, old_temps_host, N * sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
	
    float result[N];
    fclose(fopen("output_1d.txt", "w"));	

    for (int t = 0; t < 100; t++) {
        heat_update<<<blocksPerGrid, threadsPerBlock>>>(old_temps_d, new_temps_d, N, THERMAL_DIFFUSIVITY, DISTANCE_BETWEEN_CELLS);
        cudaDeviceSynchronize();

	cudaMemcpy(result, old_temps_d, N * sizeof(float), cudaMemcpyDeviceToHost);
        write_to_output_1d("output_1d.txt", result, N);

        float *temp = old_temps_d;
        old_temps_d = new_temps_d;
        new_temps_d = temp;

    }

    cudaMemcpy(result, old_temps_d, N * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Final temperatures:\n");
    for (int i = 0; i < N; i++) {
        printf("T[%d] = %f\n", i, result[i]);
    }

    cudaFree(old_temps_d);
    cudaFree(new_temps_d);
    return 0;
}
