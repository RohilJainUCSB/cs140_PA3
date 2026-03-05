Last name of Student 1: Gehlot
First name of Student 1: Sharanya
Email of Student 1: sgehlot@ucsb.edu
GradeScope account name of Student 1: Sharanya Gehlot
Last name of Student 2: Jain
First name of Student 2: Rohil
Email of Student 2: rohiljain@ucsb.edu
GradeScope account name of Student 2: Rohil Jain

----------------------------------------------------------------------------
Report for Question 1
----------------------------------------------------------------------------

List your code change for this question:

1. mult_vec() GPU kernel:
   - Computes unique thread ID: idx = blockIdx.x * blockDim.x + threadIdx.x
   - Each thread handles rows_per_thread = N / (num_blocks * threads_per_block) rows
   - For each owned row_index, computes: y[row_index] = d[row_index] + sum_j(A[row_index][j] * x[j])
   - Added bounds check (if row_index < n) to prevent out-of-bounds memory access
   - Computes diff[row_index] = |y[row_index] - x[row_index]| for convergence check

2. it_mult_vec() host function:
   - Allocates device memory for A, x, y, d, diff using cudaMalloc
   - Copies data from host to device using cudaMemcpy (HostToDevice)
   - In each iteration k, launches: mult_vec<<<num_blocks, threads_per_block>>>()
   - After each kernel launch, copies diff back to host and checks convergence threshold
   - Swaps x_d and y_d device pointers so next iteration reads the latest solution
   - Copies final y back to host after loop completes

Parallel time for n=4K, t=1K,  4x128  threads:  3.260764 seconds
Parallel time for n=4K, t=1K,  8x128  threads:  1.630125 seconds
Parallel time for n=4K, t=1K,  16x128 threads:  0.848297 seconds
Parallel time for n=4K, t=1K,  32x128 threads:  0.461755 seconds

Do you see a trend of speedup improvement with more threads?
Yes, there is a clear speedup trend as the number of threads increases.
Doubling the number of blocks roughly halves the execution time:
  4x128 -> 8x128:   ~2.00x speedup
  8x128 -> 16x128:  ~1.92x speedup
  16x128 -> 32x128: ~1.84x speedup
Overall from 4x128 to 32x128 (8x more threads): ~7.07x speedup.

This is expected because the Jacobi method is embarrassingly parallel —
each thread independently computes a block of rows y[i] = d[i] + A[i,:]*x
with no data dependencies between threads. More threads means more rows
computed in parallel per iteration, directly reducing wall-clock time.
The speedup is close to linear, indicating good GPU utilization.

----------------------------------------------------------------------------
Report for Question 2
----------------------------------------------------------------------------

List your code change for this question:

1. mult_vec_async() GPU kernel:
   - Computes unique thread ID: idx = blockIdx.x * blockDim.x + threadIdx.x
   - Initializes y[row_index] = x[row_index] for all owned rows before iterations
   - Runs num_async_iter (r=5) inner iterations WITHOUT inter-thread synchronization
   - Each inner iteration: y[row_index] = d[row_index] + sum_j(A[row_index][j] * y[j])
     (uses y instead of x — reads latest available values written by any thread, Gauss-Seidel style)
   - After all inner iterations, computes diff[row_index] = |x[row_index] - y[row_index]|

2. it_mult_vec() host function (async branch):
   - Launches mult_vec_async<<<num_blocks, threads_per_block>>>() each outer iteration
   - Increments k by NUM_ASYNC_ITER (r=5) per kernel launch since each call does r iterations
   - Checks convergence after each kernel call by copying diff back to host

Default number of asynchronous iterations per batch: r = 5 (NUM_ASYNC_ITER in it_mult_vec.h)

n=4K, t=1K, 8x128 threads, Async Gauss-Seidel:
  Parallel time:              0.040352 seconds
  Actual iterations executed: 15 (early convergence, exit before t=1024)

n=4K, t=1K, 32x128 threads, Async Gauss-Seidel:
  Parallel time:              0.474919 seconds
  Actual iterations executed: 1025 (did not converge within t=1024, error=0.001747)

Is the number of iterations executed bigger or smaller than sequential Gauss-Seidel?
The answer depends on the thread configuration:
- 8x128 threads converges in only 15 iterations — FEWER than sequential Gauss-Seidel.
- 32x128 threads does NOT converge within 1024 iterations — MORE than sequential Gauss-Seidel.

Explanation based on running trace:
In sequential Gauss-Seidel, each element y[i] is updated using the most recently
computed values in strict order (i=0,1,...,n-1), which maximizes information
propagation and leads to fast convergence.

With 8x128 threads, each thread handles more rows (rows_per_thread = 4096/1024 = 4).
Within each thread, rows are updated sequentially, preserving much of the Gauss-Seidel
ordering. Async reads of y[] also benefit from recently updated values by other threads.
This results in faster convergence than sequential GS — only 15 iterations needed.

With 32x128 threads, each thread handles fewer rows (rows_per_thread = 4096/4096 = 1).
With only 1 row per thread, there is minimal sequential ordering within any thread.
Threads run asynchronously and may read stale values of y[], weakening the Gauss-Seidel
effect significantly. This causes slow convergence, and the method fails to converge
within the 1024 iteration budget, taking much longer (0.47s) than the 8x128 case (0.04s).

Therefore, more threads do NOT always yield more time reduction for async Gauss-Seidel.
Convergence quality degrades with more threads due to loss of sequential ordering,
which can completely offset the benefit of increased parallelism.

Output trace of it_mult_vec_test.cu on Expanse GPU (Q1 and Q2):
----------------------------------------------------------------
>>>>>>>>>>>>>>>>>>>>>>>>>
Start running itmv tests.
>>>>>>>>>>>>>>>>>>>>>>>>>
Test 1:n=4, t=1, 1x2 threads:
With totally 1*2 threads, matrix size being 4, t being 1
Time cost in seconds: 0.123807
Final error (|y-x|): 1.750000.
# of iterations executed: 1.
Final y[0]=1.750000. y[n-1]=1.750000

Test 2:n=4, t=2, 1x2 threads:
With totally 1*2 threads, matrix size being 4, t being 2
Time cost in seconds: 0.000251
Final error (|y-x|): 1.312500.
# of iterations executed: 2.
Final y[0]=0.437500. y[n-1]=0.437500

Test 3:n=8, t=1, 1x2 threads:
With totally 1*2 threads, matrix size being 8, t being 1
Time cost in seconds: 0.000236
Final error (|y-x|): 1.875000.
# of iterations executed: 1.
Final y[0]=1.875000. y[n-1]=1.875000

Test 4:n=8, t=2, 1x2 threads:
With totally 1*2 threads, matrix size being 8, t being 2
Time cost in seconds: 0.000238
Final error (|y-x|): 1.640625.
# of iterations executed: 2.
Final y[0]=0.234375. y[n-1]=0.234375

Test 8a:n=4, t=1, 1x1 threads/Gauss-Seidel:
With totally 1*1 threads, matrix size being 4, t being 1
Time cost in seconds: 0.000235
Final error (|y-x|): 1.000193.
# of iterations executed: 5.
Final y[0]=1.000089. y[n-1]=1.000193

Test 8b:n=4, t=2, 1x1 threads/Gauss-Seidel:
With totally 1*1 threads, matrix size being 4, t being 2
Time cost in seconds: 0.000237
Final error (|y-x|): 1.000193.
# of iterations executed: 5.
Final y[0]=1.000089. y[n-1]=1.000193

Test 8c:n=8, t=1, 1x1 threads/Gauss-Seidel:
With totally 1*1 threads, matrix size being 8, t being 1
Time cost in seconds: 0.000229
Final error (|y-x|): 1.001155.
# of iterations executed: 5.
Final y[0]=1.001155. y[n-1]=0.999790

Test 8d:n=8, t=2, 1x1 threads/Gauss-Seidel:
With totally 1*1 threads, matrix size being 8, t being 2
Time cost in seconds: 0.000222
Final error (|y-x|): 1.001155.
# of iterations executed: 5.
Final y[0]=1.001155. y[n-1]=0.999790

Test 9: n=4K t=1K 32x128 threads:
With totally 32*128 threads, matrix size being 4096, t being 1024
Time cost in seconds: 0.461755
Final error (|y-x|): 1.500505.
# of iterations executed: 1024.
Final y[0]=0.249800. y[n-1]=0.249800

Test 9a: n=4K t=1K 16x128 threads:
With totally 16*128 threads, matrix size being 4096, t being 1024
Time cost in seconds: 0.848297
Final error (|y-x|): 1.500505.
# of iterations executed: 1024.
Final y[0]=0.249800. y[n-1]=0.249800

Test 9b: n=4K t=1K 8x128 threads:
With totally 8*128 threads, matrix size being 4096, t being 1024
Time cost in seconds: 1.630125
Final error (|y-x|): 1.500505.
# of iterations executed: 1024.
Final y[0]=0.249800. y[n-1]=0.249800

Test 9c: n=4K t=1K 4x128 threads:
With totally 4*128 threads, matrix size being 4096, t being 1024
Time cost in seconds: 3.260764
Final error (|y-x|): 1.500505.
# of iterations executed: 1024.
Final y[0]=0.249800. y[n-1]=0.249800

Test 11: n=4K t=1K 32x128 threads/Async:
With totally 32*128 threads, matrix size being 4096, t being 1024
Time cost in seconds: 0.474919
Final error (|y-x|): 0.001747.
# of iterations executed: 1025.
Final y[0]=1.000801. y[n-1]=1.000824

Test 11a: n=4K t=1K 8x128 threads/Async:
With totally 8*128 threads, matrix size being 4096, t being 1024
Time cost in seconds: 0.040352
Final error (|y-x|): 0.000000.
# of iterations executed: 15.
Early exit due to convergence, even asked for 1024 iterations.
Asynchronous code actually runs 15 iterations.
Final y[0]=1.000000. y[n-1]=1.000000

Summary: Failed 0 out of 14 tests

----------------------------------------------------------------------------
Report for Question 3
----------------------------------------------------------------------------

Solution to call cublasDgemm() in Method 1:

  cublasDgemm(handle,
              CUBLAS_OP_N, CUBLAS_OP_N,
              N, N, N,
              &alpha,
              d_A, N,
              d_B, N,
              &beta,
              d_C, N);

Where alpha = 1.0 and beta = 0.0 to compute C = alpha*A*B + beta*C = A*B.
CUBLAS_OP_N specifies no transpose for both A and B.
Leading dimension for all matrices is N (column-major storage).

Latency and GFLOPs of 3 implementations (V100 GPU):

N=50:
  1. cuBLAS dgemm (optimized):   Latency=0.002 ms,    GFLOPs=122.070
  2. cuBLAS dgemv loop:          Latency=0.155 ms,    GFLOPs=1.617
  3. Naive GEMM (3 loops):       Latency=0.008 ms,    GFLOPs=30.399

N=200:
  1. cuBLAS dgemm (optimized):   Latency=0.002 ms,    GFLOPs=6410.257
  2. cuBLAS dgemv loop:          Latency=3.346 ms,    GFLOPs=4.781
  3. Naive GEMM (3 loops):       Latency=0.030 ms,    GFLOPs=538.793

N=800:
  1. cuBLAS dgemm (optimized):   Latency=0.003 ms,    GFLOPs=405063.303
  2. cuBLAS dgemv loop:          Latency=13.198 ms,   GFLOPs=77.586
  3. Naive GEMM (3 loops):       Latency=1.080 ms,    GFLOPs=947.867

N=1600:
  1. cuBLAS dgemm (optimized):   Latency=0.003 ms,    GFLOPs=3240506.426
  2. cuBLAS dgemv loop:          Latency=46.288 ms,   GFLOPs=176.979
  3. Naive GEMM (3 loops):       Latency=8.392 ms,    GFLOPs=976.205

Number of CUDA threads used in Method 3 (Naive GEMM) for N=1600:
  Using 16x16 thread blocks: grid = ceil(1600/16) x ceil(1600/16) = 100x100 blocks
  Total CUDA threads = 100 * 100 * 16 * 16 = 2,560,000 threads

Highest GFLOPs observed with V100 (this question, N=1600):
  cuBLAS dgemm: 3,240,506 GFLOPs (~3.24 TFLOPS)

Speedup of V100 over CPU host:
  Ratio = V100 GFLOPs / PA2 MKL GFLOPs
        = 3,240,506 / 79.45
        = ~40,789x speedup of V100 GPU over single-threaded CPU MKL DGEMM
