#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define THERMAL_DIFFUSIVITY 100.0f
#define DISTANCE_BETWEEN_CELLS 100.0f

__global__ void heat_update(float *old_temps, float *new_temps, int N, float alpha, float dx, int width) {

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= N) return;

    int row = i / width;
    int col = i % width;

    float curr_temp = old_temps[i];
    float laplacian = 0.0f;

    if (row > 0){
        laplacian += old_temps[i - width];
    }

    if (row < width - 1){
        laplacian += old_temps[i + width];
    }


    if (col > 0){
        laplacian += old_temps[i - 1];
    }

    if (col < width - 1){
        laplacian += old_temps[i + 1];
    }

    laplacian -= 4.0f * curr_temp;

    new_temps[i] = curr_temp + alpha * (laplacian / (dx * dx));
}

int main(void) {

    const int N = 9; // width and height of square grid
    const int width = sqrt(N);
    float old_temps_host[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};

    float *old_temps_d, *new_temps_d;
    cudaMalloc((void **)&old_temps_d, N * sizeof(float));
    cudaMalloc((void **)&new_temps_d, N * sizeof(float));

    cudaMemcpy(old_temps_d, old_temps_host, N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blocksPerGrid((N + 15) / 16, (N + 15) / 16);
    dim3 threadsPerBlock(16, 16);

    for (int t = 0; t < 100; t++) {
        heat_update<<<blocksPerGrid, threadsPerBlock>>>(old_temps_d, new_temps_d, N, THERMAL_DIFFUSIVITY, DISTANCE_BETWEEN_CELLS, );
        cudaDeviceSynchronize();

        float *temp = old_temps_d;
        old_temps_d = new_temps_d;
        new_temps_d = temp;
    }

    float result[N];
    cudaMemcpy(result, old_temps_d, N * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Final temperatures:\n");
    for (int i = 0; i < N; i++) {
        printf("T[%d] = %f\n", i, result[i]);
    }

    cudaFree(old_temps_d);
    cudaFree(new_temps_d);

    return 0;
}
