#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// Matrix dimensions and block size configuration
const int BLOCK_SIZE = 20; 

/**
 * @brief Naive square matrix multiplication kernel for C = A * B.
 * * Uses Column-Major indexing: M[i][j] is stored at M[i + j * N].
 * This is an unoptimized kernel intended as a baseline comparison.
 */
__global__ void naive_matrix_mult_kernel(const double *A, const double *B, double *C, int size_N) {
    // Row index i and column index j for matrix C
    int i = blockIdx.x * blockDim.x + threadIdx.x; // Row index
    int j = blockIdx.y * blockDim.y + threadIdx.y; // Column index

    if (i < size_N && j < size_N) {
        double sum = 0.0f;
        
        // Inner product calculation (dot product of A's row i and B's column j)
        for (int k = 0; k < size_N; k++) {
            // A[i][k] in CM: A[i + k * size_N]
            // B[k][j] in CM: B[k + j * size_N]
            sum += A[i + k * size_N] * B[k + j * size_N];
        }
        
        // C[i][j] in CM: C[i + j * size_N]
        C[i + j * size_N] = sum;
    }
}


// --- Error Handling and Utility Functions ---

/**
 * @brief Checks for CUDA errors and reports if one occurred.
 */
void handle_cuda_error(cudaError_t err, const char* func, const char* file, int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s in %s at %s:%d\n", cudaGetErrorString(err), func, file, line);
        exit(EXIT_FAILURE);
    }
}

/**
 * @brief Checks for cuBLAS errors and reports if one occurred.
 */
void handle_cublas_error(cublasStatus_t status, const char* func, const char* file, int line) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "CUBLAS Error: %d in %s at %s:%d\n", status, func, file, line);
        exit(EXIT_FAILURE);
    }
}

/**
 * @brief Calculates GFLOPS based on execution time and matrix size.
 */
double calculate_gflops(double time_ms, int n) {
    // FLOPS = 2 * N^3 (multiplications and additions)
    double flops = 2.0 * n * n * n;
    double time_s = time_ms / 1000.0;
    return (flops / time_s) / 1e9;
}


// --- Implementation 3: Naive Custom Kernel (CM) ---

/**
 * @brief Executes the custom naive GEMM kernel and measures performance.
 */
void run_naive_gemm(const double *d_A, const double *d_B, double *d_C, int size_N, double *latency_ms, double *gflops) {
    cudaEvent_t start, stop;
    handle_cuda_error(cudaEventCreate(&start), "cudaEventCreate(&start)", __FILE__, __LINE__);
    handle_cuda_error(cudaEventCreate(&stop), "cudaEventCreate(&stop)", __FILE__, __LINE__);

    // Determine grid and block dimensions for the kernel launch
    dim3 gridDim((size_N + BLOCK_SIZE - 1) / BLOCK_SIZE, (size_N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    
    // Warm-up run
    naive_matrix_mult_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, size_N);
    handle_cuda_error(cudaGetLastError(), "cudaGetLastError()", __FILE__, __LINE__);
    handle_cuda_error(cudaDeviceSynchronize(), "cudaDeviceSynchronize()", __FILE__, __LINE__);

    // Main run with timing
    handle_cuda_error(cudaEventRecord(start, 0), "cudaEventRecord(start, 0)", __FILE__, __LINE__);
    naive_matrix_mult_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, size_N);
    handle_cuda_error(cudaGetLastError(), "cudaGetLastError()", __FILE__, __LINE__);
    handle_cuda_error(cudaEventRecord(stop, 0), "cudaEventRecord(stop, 0)", __FILE__, __LINE__);
    handle_cuda_error(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)", __FILE__, __LINE__);

    float ms = 0.0f;
    handle_cuda_error(cudaEventElapsedTime(&ms, start, stop), "cudaEventElapsedTime(&ms, start, stop)", __FILE__, __LINE__);
    
    *latency_ms = (double)ms;
    *gflops = calculate_gflops(*latency_ms, size_N);

    handle_cuda_error(cudaEventDestroy(start), "cudaEventDestroy(start)", __FILE__, __LINE__);
    handle_cuda_error(cudaEventDestroy(stop), "cudaEventDestroy(stop)", __FILE__, __LINE__);
}


// --- Implementation 1: cuBLAS dgemm (Optimized GEMM, CM Native) ---

/**
 * @brief Executes matrix multiplication using the highly optimized cuBLAS dgemm function.
 * * Since data is Column-Major (CM), we use the standard dgemm call: C = A * B.
 * CUBLAS_OP_N indicates the matrix is treated as Non-Transposed.
 */
void run_cublas_dgemm(cublasHandle_t handle, const double *d_A, const double *d_B, double *d_C, int size_N, double *latency_ms, double *gflops) {
    cudaEvent_t start, stop;
    handle_cuda_error(cudaEventCreate(&start), "cudaEventCreate(&start)", __FILE__, __LINE__);
    handle_cuda_error(cudaEventCreate(&stop), "cudaEventCreate(&stop)", __FILE__, __LINE__);

    const double alpha = 1.0f;
    const double beta = 0.0f;

    // A and B are CM. cuBLAS natively uses CM, so we use CUBLAS_OP_N (No Transposition).
    // The order of matrices is standard: C = alpha * A * B + beta * C
    
    // Warm-up run (This is an optional action)
    // Your solution to make a warm-up call to cublasDgemm  in the next line.

    // Main run with timing
    handle_cuda_error(cudaEventRecord(start, 0), "cudaEventRecord(start, 0)", __FILE__, __LINE__);

    // Your solution to make a cublasDgemm call in the next line
    handle_cublas_error( (cublasStatus_t) NULL,
                 "cublasDgemm (Main)", __FILE__, __LINE__);
    handle_cuda_error(cudaEventRecord(stop, 0), "cudaEventRecord(stop, 0)", __FILE__, __LINE__);
    handle_cuda_error(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)", __FILE__, __LINE__);

    float ms = 0.0f;
    handle_cuda_error(cudaEventElapsedTime(&ms, start, stop), "cudaEventElapsedTime(&ms, start, stop)", __FILE__, __LINE__);

    *latency_ms = (double)ms;
    *gflops = calculate_gflops(*latency_ms, size_N);
    
    handle_cuda_error(cudaEventDestroy(start), "cudaEventDestroy(start)", __FILE__, __LINE__);
    handle_cuda_error(cudaEventDestroy(stop), "cudaEventDestroy(stop)", __FILE__, __LINE__);
}


// --- Implementation 2: cuBLAS dgemv in a Host Loop (CM) ---

/**
 * @brief Executes matrix multiplication by repeatedly calling the cuBLAS matrix-vector product (dgemv).
 * * This simulates GEMM inefficiently by launching N separate kernels from the host CPU.
 * For C = A * B (CM): C's j-th column is A * (B's j-th column).
 * We iterate over the columns (j) and compute the vector result using dgemv: c_j = A * b_j.
 */
void run_cublas_dgemv_loop(cublasHandle_t handle, const double *d_A, const double *d_B, double *d_C, int size_N, double *latency_ms, double *gflops) {
    cudaEvent_t start, stop;
    handle_cuda_error(cudaEventCreate(&start), "cudaEventCreate(&start)", __FILE__, __LINE__);
    handle_cuda_error(cudaEventCreate(&stop), "cudaEventCreate(&stop)", __FILE__, __LINE__);

    const double alpha = 1.0f;
    const double beta = 0.0f;
    
    // We iterate over the columns (j) of B and C
    for (int j = 0; j < size_N; j++) {
        // Compute C_col_j = A * B_col_j
        // d_A is the CM matrix A (CUBLAS_OP_N)
        // d_B + j * size_N is the j-th column of B (the input vector)
        // d_C + j * size_N is the j-th column of C (the output vector)
        handle_cublas_error(cublasDgemv(handle, CUBLAS_OP_N, size_N, size_N, &alpha, d_A, size_N, d_B + j * size_N, 1, &beta, d_C + j * size_N, 1), "cublasDgemv (Warmup)", __FILE__, __LINE__);
    }
    handle_cuda_error(cudaDeviceSynchronize(), "cudaDeviceSynchronize()", __FILE__, __LINE__);

    // Main run with timing
    handle_cuda_error(cudaEventRecord(start, 0), "cudaEventRecord(start, 0)", __FILE__, __LINE__);
    for (int j = 0; j < size_N; j++) {
        handle_cublas_error(cublasDgemv(handle, CUBLAS_OP_N, size_N, size_N, &alpha, d_A, size_N, d_B + j * size_N, 1, &beta, d_C + j * size_N, 1), "cublasDgemv (Main)", __FILE__, __LINE__);
    }
    handle_cuda_error(cudaEventRecord(stop, 0), "cudaEventRecord(stop, 0)", __FILE__, __LINE__);
    handle_cuda_error(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)", __FILE__, __LINE__);

    float ms = 0.0f;
    handle_cuda_error(cudaEventElapsedTime(&ms, start, stop), "cudaEventElapsedTime(&ms, start, stop)", __FILE__, __LINE__);

    *latency_ms = (double)ms;
    *gflops = calculate_gflops(*latency_ms, size_N);

    handle_cuda_error(cudaEventDestroy(start), "cudaEventDestroy(start)", __FILE__, __LINE__);
    handle_cuda_error(cudaEventDestroy(stop), "cudaEventDestroy(stop)", __FILE__, __LINE__);
}


// --- Main Function: Setup, Execution, and Cleanup ---

// Matrix dimension N 
int compare( const int N) {
    printf("CUDA matrix-matrix multiply (%dx%d matrices) with column major\n", N, N);

    const int matrix_size = N * N;
    
    // Initialize cuBLAS library handle
    cublasHandle_t cublas_handle;
    handle_cublas_error(cublasCreate(&cublas_handle), "cublasCreate(&cublas_handle)", __FILE__, __LINE__);

    // Host memory allocation
    double *h_A = (double*)malloc(matrix_size * sizeof(double));
    double *h_B = (double*)malloc(matrix_size * sizeof(double));
    double *h_C= (double*)malloc(matrix_size * sizeof(double));

    // Initialize input matrices with random data
    // Data is initialized in CM (column-major) fashion, just like how it will be stored on the device
    srand(time(NULL));
    for (int i = 0; i < matrix_size; i++) {
        h_A[i] = (double)rand() / (double)RAND_MAX; 
        h_B[i] = (double)rand() / (double)RAND_MAX; 
        h_C[i] = 0.0f; // Unused but good practice
    }

    // Device memory allocation
    double *d_A, *d_B, *d_C;
    handle_cuda_error(cudaMalloc((void**)&d_A, matrix_size * sizeof(double)), "cudaMalloc((void**)&d_A, ...)", __FILE__, __LINE__);
    handle_cuda_error(cudaMalloc((void**)&d_B, matrix_size * sizeof(double)), "cudaMalloc((void**)&d_B, ...)", __FILE__, __LINE__);
    handle_cuda_error(cudaMalloc((void**)&d_C, matrix_size * sizeof(double)), "cudaMalloc((void**)&d_C, ...)", __FILE__, __LINE__);

    // Copy CM data from host to device
    handle_cuda_error(cudaMemcpy(d_A, h_A, matrix_size * sizeof(double), cudaMemcpyHostToDevice), "cudaMemcpy(d_A, h_A, ...)", __FILE__, __LINE__);
    handle_cuda_error(cudaMemcpy(d_B, h_B, matrix_size * sizeof(double), cudaMemcpyHostToDevice), "cudaMemcpy(d_B, h_B, ...)", __FILE__, __LINE__);
    
    double latency_naive, gflops_naive;
    double latency_dgemm, gflops_dgemm;
    double latency_dgemv, gflops_dgemv;


    // --- Run 1: cuBLAS dgemm ---
    run_cublas_dgemm(cublas_handle, d_A, d_B, d_C, N, &latency_dgemm, &gflops_dgemm);
    handle_cuda_error(cudaMemcpy(h_C, d_C, matrix_size * sizeof(double), cudaMemcpyDeviceToHost), "cudaMemcpy(h_C, d_C, ...)", __FILE__, __LINE__);
    double dgemm_val = h_C[N / 2 * N + N / 2]; // Mid-point
    printf("1. cuBLAS dgemm (optimized library) ");
    printf("   Latency: %.3f ms", latency_dgemm);
    printf("   GFLOPS:  %.3f\n", gflops_dgemm);

    // --- Run 2: cuBLAS dgemv loop ---
    printf("2. cuBLAS dgemv in a loop for GEMM  ");
    // Clear output matrix d_C before running the dgemv loop
    handle_cuda_error(cudaMemset(d_C, 0, matrix_size * sizeof(double)), "cudaMemset(d_C, 0, ...)", __FILE__, __LINE__);
    run_cublas_dgemv_loop(cublas_handle, d_A, d_B, d_C, N, &latency_dgemv, &gflops_dgemv);
    handle_cuda_error(cudaMemcpy(h_C, d_C, matrix_size * sizeof(double), cudaMemcpyDeviceToHost), "cudaMemcpy(h_C, d_C, ...)", __FILE__, __LINE__);
    double dgemv_val = h_C[N / 2 * N + N / 2]; // Mid-point
    printf("   Latency: %.3f ms", latency_dgemv);
    printf("   GFLOPS:  %.3f\n", gflops_dgemv);
    
    // --- Run 3: Naive Kernel ---
    printf("3. Naive GEMM (parallelized 3 loops)");
    handle_cuda_error(cudaMemset(d_C, 0, matrix_size * sizeof(double)), "cudaMemset(d_C, 0, ...)", __FILE__, __LINE__);
    run_naive_gemm(d_A, d_B, d_C, N, &latency_naive, &gflops_naive);
    handle_cuda_error(cudaMemcpy(h_C, d_C, matrix_size * sizeof(double), cudaMemcpyDeviceToHost), "cudaMemcpy(h_C, d_C, ...)", __FILE__, __LINE__);
    double naive_val = h_C[N/2 * N + N/2]; // Mid-point
    printf("   Latency: %.3f ms", latency_naive);
    printf("   GFLOPS:  %.3f\n", gflops_naive);


    // Compare one element to ensure correctness (DGEMM is the ground truth)
    if(fabs(dgemm_val - dgemv_val) >0.00001 || fabs(dgemm_val - naive_val) >0.00001)
      printf("\nError! Unequal mid-points:");
    else
      printf("\nMid-point verification looks OK:");
    printf(" DGEMM=%.4f, DGEMV=%.4f, Naive=%.4f\n\n", dgemm_val, dgemv_val, naive_val);

    // Cleanup
    handle_cublas_error(cublasDestroy(cublas_handle), "cublasDestroy(cublas_handle)", __FILE__, __LINE__);

    handle_cuda_error(cudaFree(d_A), "cudaFree(d_A)", __FILE__, __LINE__);
    handle_cuda_error(cudaFree(d_B), "cudaFree(d_B)", __FILE__, __LINE__);
    handle_cuda_error(cudaFree(d_C), "cudaFree(d_C)", __FILE__, __LINE__);

    free(h_A);
    free(h_B);
    free(h_C);

    //printf("Comparison finished. GFLOPS is the best indicator of performance efficiency.\n");
    
    return 0;
}

int main(){
    compare(50);
    compare(200);
    compare(800);
    compare(1600);
    compare(3200);
}
