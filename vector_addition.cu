#include <iostream>
#include <cmath>
#include <omp.h>
#include <cuda_runtime.h>

const int N = 1048576;

__global__ void add_vectors(const float* A, const float* B, float* C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    float *h_A = new float[N];
    float *h_B = new float[N];
    float *h_C_CPU = new float[N];
    float *h_C_GPU = new float[N];

    for (int i = 0; i < N; i++) {
        h_A[i] = static_cast<float>(i) * 0.5f;
        h_B[i] = static_cast<float>(i) * 2.0f;
    }

    std::cout << "====================================================\n";
    std::cout << " Monitoreo de Suma de Vectores - Dimension N = " << N << "\n";
    std::cout << "====================================================\n\n";

    double start_cpu = omp_get_wtime();
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        h_C_CPU[i] = h_A[i] + h_B[i];
    }
    double end_cpu = omp_get_wtime();
    double t_cpu = end_cpu - start_cpu;
    std::cout << "[CPU] Tiempo consumido por OpenMP: " << t_cpu << " segundos.\n";

    double start_gpu = omp_get_wtime();
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, N * sizeof(float));
    cudaMalloc(&d_B, N * sizeof(float));
    cudaMalloc(&d_C, N * sizeof(float));

    cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    add_vectors<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C_GPU, d_C, N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    double end_gpu = omp_get_wtime();
    double t_gpu = end_gpu - start_gpu;
    std::cout << "[GPU] Tiempo global registrado en CUDA: " << t_gpu << " segundos.\n\n";

    bool success = true;
    for (int i = 0; i < N; i++) {
        if (std::fabs(h_C_GPU[i] - h_C_CPU[i]) > 1e-4) {
            success = false;
            break;
        }
    }

    if (success) {
        std::cout << ">> VALIDACION CORRECTA: Los arreglos de salida coinciden.\n";
    } else {
        std::cout << ">> ERROR: Divergencia numerica entre los resultados.\n";
    }

    double speedup = t_cpu / t_gpu;
    std::cout << ">> Factor de Aceleracion (Speedup): " << speedup << "x\n\n";

    delete[] h_A;
    delete[] h_B;
    delete[] h_C_CPU;
    delete[] h_C_GPU;

    return 0;
}
