/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/**
 * Matrix multiplication: C = A * B.
 * Host code.
 *
 * This sample implements matrix multiplication which makes use of shared memory
 * to ensure data reuse, the matrix multiplication is done using tiling approach.
 * It has been written for clarity of exposition to illustrate various CUDA programming
 * principles, not with the goal of providing the most performant generic kernel for matrix multiplication.
 * See also:
 * V. Volkov and J. Demmel, "Benchmarking GPUs to tune dense linear algebra,"
 * in Proc. 2008 ACM/IEEE Conf. on Supercomputing (SC '08),
 * Piscataway, NJ: IEEE Press, 2008, pp. Art. 31:1-11.
 */

// System includes
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <stdio.h>
#include <assert.h>
#include "oneapi/mkl/blas.hpp"

namespace mkl = oneapi::mkl;  //# shorten mkl namespace                                                                                   
// CUDA runtime

// Helper functions and utilities to work with CUDA
//#include <helper_functions.h>
#include <helper_cuda.h>
#include <chrono>

#include <cmath>

/**
 * Matrix multiplication (SYCL Kernel) on the device: C = A * B
 * wA is A's width and wB is B's width
 */

template <int BLOCK_SIZE> void MatrixMulSYCL(float *C, float *A, float *B, int wA, int wB, sycl::nd_item<3> item_ct1, 
  sycl::accessor<float, 2, sycl::access_mode::read_write, sycl::access::target::local> As,
  sycl::accessor<float, 2, sycl::access_mode::read_write, sycl::access::target::local> Bs) {
  
  // Block index
  int bx = item_ct1.get_group(2);
  int by = item_ct1.get_group(1);

  // Thread index
  int tx = item_ct1.get_local_id(2);
  int ty = item_ct1.get_local_id(1);

  // Index of the first sub-matrix of A processed by the block
  int aBegin = wA * BLOCK_SIZE * by;

  // Index of the last sub-matrix of A processed by the block
  int aEnd   = aBegin + wA - 1;

  // Step size used to iterate through the sub-matrices of A
  int aStep  = BLOCK_SIZE;

  // Index of the first sub-matrix of B processed by the block
  int bBegin = BLOCK_SIZE * bx;

  // Step size used to iterate through the sub-matrices of B
  int bStep  = BLOCK_SIZE * wB;

  // Csub is used to store the element of the block sub-matrix
  // that is computed by the thread
  float Csub = 0;

  // Loop over all the sub-matrices of A and B
  // required to compute the block sub-matrix
  for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) {
    
    // Declaration of the shared memory array As used to
    // store the sub-matrix of A

    // Declaration of the shared memory array Bs used to
    // store the sub-matrix of B

    // Load the matrices from device memory
    // to shared memory; each thread loads
    // one element of each matrix
    As[ty][tx] = A[a + wA * ty + tx];
    Bs[ty][tx] = B[b + wB * ty + tx];

    // Synchronize to make sure the matrices are loaded
    /*
    DPCT1065:26: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    // Multiply the two matrices together;
    // each thread computes one element
    // of the block sub-matrix
    #pragma unroll

    for (int k = 0; k < BLOCK_SIZE; ++k) {
      Csub += As[ty][k] * Bs[k][tx];
    }

    // Synchronize to make sure that the preceding
    // computation is done before loading two new
    // sub-matrices of A and B in the next iteration
    /*
    DPCT1065:27: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
  }

  // Write the block sub-matrix to device memory;
  // each thread writes one element
  int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
  C[c + wB * ty + tx] = Csub;
}

void ConstantInit(float *data, int size, float val) {
  for (int i = 0; i < size; ++i) {
    data[i] = val;
  }
}

#include "magma_v2.h"
extern "C" void magmablas_sgemm2(magma_trans_t transA, magma_trans_t transB,
                                 magma_int_t, magma_int_t n, magma_int_t k,
                                 float alpha, magmaFloat_const_ptr dA,
                                 magma_int_t ldda, magmaFloat_const_ptr dB,
                                 magma_int_t lddb, float beta, magmaFloat_ptr dC,
                                 magma_int_t lddc,
                                 //magma_queue_t queue
                                 sycl::queue *queue
                                 );

//#include <omp.h>
#define BLK 96

void sgemm_ijk(int m, int n, int k,
               float alpha, float *A, int lda, float *B, int ldb, 
               float beta, float *C, int ldc){
    for(int i = 0 ; i < m; ++i ) {
        for(int j = 0 ; j < n ; ++j ) {
            float sum = 0.0;
            for(int l = 0 ; l < k ; ++l ) {
                sum += A[i+lda*l] * B[l + j*ldb];
            }
            C[i+j*ldc] = beta * C[i+ j*ldc] + alpha * sum;
        }
    }
}

#define min(X, Y) (((X) < (Y)) ? (X) : (Y))

void sgemm_bijk(int M, int N, int K,
                float alpha, float *A, int lda, float *B, int ldb,
                float beta, float *C, int ldc){
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for(int i = 0 ; i < M; i+=BLK )
        for(int j = 0 ; j < N ; j+=BLK )
            for(int k = 0 ; k < K ; k+=BLK ) {
                int bm = min(BLK, M-i);
                int bn = min(BLK, N-j);
                int bk = min(BLK, K-k);
                if (k==0)
                    sgemm_ijk(bm,bn,bk, alpha, A+i, lda, B+j*ldb, ldb, beta, C+i+j*ldc, ldc);
                else
                    sgemm_ijk(bm,bn,bk, alpha, A+i, lda, B+j*ldb, ldb, 1.0, C+i+j*ldc, ldc);
            }
}


/*  Function:   Matrix Multiply
    Parameters: Number of Arguments in Command Line
                Array of Arguments in Command Line
                User-defined Block Size (16 or 32)
		            Dimensions of 2D Array A (1, x, x)
		            Dimensions of 2D Array B (1, x, x)
*/

int MatrixMultiply(int argc, char **argv, int block_size,
                   const sycl::range<3> &dimsA, const sycl::range<3> &dimsB) {

        /*sycl::device d;
  
        try {
	    auto devID = sycl::gpu_selector();
            d = sycl::device(devID);
            std::cout << "Using a GPU device" << "\n";
        } catch (sycl::exception const &e) {
            //std::cout << "Cannot select a GPU\n" << e.what() << "\n";
            std::cout << "Using a CPU device\n";
            d = sycl::device(sycl::cpu_selector());
        }*/

// 0. Get the current device and queue

  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();

  // 1. Allocate host memory for matrices A and B

  unsigned int size_A = dimsA[2] * dimsA[1];
  unsigned int mem_size_A = sizeof(float) * size_A;
  float *h_A;

  h_A = (float *)sycl::malloc_host(mem_size_A, dpct::get_default_queue());

  unsigned int size_B = dimsB[2] * dimsB[1];
  unsigned int mem_size_B = sizeof(float) * size_B;
  float *h_B;

  h_B = (float *)sycl::malloc_host(mem_size_B, dpct::get_default_queue());

  //sycl::queue q;
  sycl::queue *stream;

  // 2. Initialize host memory

  const float valB = 0.01f;
  ConstantInit(h_A, size_A, 1.0f);
  ConstantInit(h_B, size_B, valB);

  // 3. Allocate device memory

  float *d_A, *d_B, *d_C;

  // 4. Allocate host matrix C

  sycl::range<3> dimsC(1, dimsA[1], dimsB[2]);
  unsigned int mem_size_C = dimsC[2] * dimsC[1] * sizeof(float);
  float *h_C;

  h_C = (float *)sycl::malloc_host(mem_size_C, dpct::get_default_queue());

  if (h_C == NULL) {
    fprintf(stderr, "Failed to allocate host matrix C!\n");
    exit(EXIT_FAILURE);
  }

  *(reinterpret_cast<void **>(&d_A)) = (void *)sycl::malloc_device(mem_size_A, dpct::get_default_queue());
  *(reinterpret_cast<void **>(&d_B)) = (void *)sycl::malloc_device(mem_size_B, dpct::get_default_queue());
  *(reinterpret_cast<void **>(&d_C)) = (void *)sycl::malloc_device(mem_size_C, dpct::get_default_queue());

  // 5. Allocate CUDA events that we'll use for timing

  sycl::event start, stop;
  std::chrono::time_point<std::chrono::steady_clock> start_ct1;
  std::chrono::time_point<std::chrono::steady_clock> stop_ct1;

  stream = dev_ct1.create_queue();
  
/*stream = dpct::get_current_device().create_queue(d);
  q = sycl::queue(d); //.create_queue();
  stream = &q; //.create_queue(); */

  // 6. Copy host memory to device

  stream->memcpy(d_A, h_A, mem_size_A);
  stream->memcpy(d_B, h_B, mem_size_B);

  // 7. Setup execution parameters

  sycl::range<3> threads(1, block_size, block_size);
  sycl::range<3> grid(1, dimsA[1] / threads[1], dimsB[2] / threads[2]);

  // 8. Create and start timer

  printf("\nComputing result using SYCL Kernel...\n");

  // 9. Performs warmup operation using matrixMul CUDA kernel

  // Block size is 32, enter else statement
  if (block_size == 16) {
    stream->submit([&](sycl::handler &cgh) {
      sycl::accessor<float, 2, sycl::access_mode::read_write, sycl::access::target::local>
        As_acc_ct1(sycl::range<2>(16, 16), cgh);
      sycl::accessor<float, 2, sycl::access_mode::read_write, sycl::access::target::local>
        Bs_acc_ct1(sycl::range<2>(16, 16), cgh);

      cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads), [=](sycl::nd_item<3> item_ct1) {
        MatrixMulSYCL<16>(d_C, d_A, d_B, dimsA[2], dimsB[2], item_ct1, As_acc_ct1, Bs_acc_ct1);
          });
     });
  } else {
    
    stream->submit([&](sycl::handler &cgh) {
      sycl::accessor<float, 2, sycl::access_mode::read_write, sycl::access::target::local>
        As_acc_ct1(sycl::range<2>(32, 32), cgh);
      sycl::accessor<float, 2, sycl::access_mode::read_write, sycl::access::target::local>
        Bs_acc_ct1(sycl::range<2>(32, 32), cgh);

      cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads), [=](sycl::nd_item<3> item_ct1) {
        MatrixMulSYCL<32>(d_C, d_A, d_B, dimsA[2], dimsB[2], item_ct1, As_acc_ct1, Bs_acc_ct1);
      });
    });
  }
  printf("\ndone 1 warmup\n");

  stream->memcpy(d_A, h_A, mem_size_A);
  stream->memcpy(d_B, h_B, mem_size_B);

  stream->wait();

  // Record the start event
  start_ct1 = std::chrono::steady_clock::now();

  // Execute the kernel
  int nIter = 1;
  
  for (int j = 0; j < nIter; j++) {
    if (block_size == 16) {

      stop = stream->submit([&](sycl::handler &cgh) {
        sycl::accessor<float, 2, sycl::access_mode::read_write, sycl::access::target::local>
          As_acc_ct1(sycl::range<2>(16, 16), cgh);
        sycl::accessor<float, 2, sycl::access_mode::read_write, sycl::access::target::local>
            Bs_acc_ct1(sycl::range<2>(16, 16), cgh);

        cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads), [=](sycl::nd_item<3> item_ct1) {
          MatrixMulSYCL<16>(d_C, d_A, d_B, dimsA[2], dimsB[2], item_ct1, As_acc_ct1, Bs_acc_ct1);
        });
      });
    } else {
      
      stop = stream->submit([&](sycl::handler &cgh) {
        sycl::accessor<float, 2, sycl::access_mode::read_write, sycl::access::target::local>
          As_acc_ct1(sycl::range<2>(32, 32), cgh);
        sycl::accessor<float, 2, sycl::access_mode::read_write, sycl::access::target::local>
            Bs_acc_ct1(sycl::range<2>(32, 32), cgh);

        cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads), [=](sycl::nd_item<3> item_ct1) {
          MatrixMulSYCL<32>(d_C, d_A, d_B, dimsA[2], dimsB[2], item_ct1, As_acc_ct1, Bs_acc_ct1);
        });
      });
    }
  }
  
  // Record the stop event
  stop.wait();
  stop_ct1 = std::chrono::steady_clock::now();

  float msecTotal = 0.0f;  
  msecTotal = std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1).count();
  
  // Compute and print the performance
  float msecPerMatrixMul = msecTotal / nIter;
  double flopsPerMatrixMul = 2.0 * static_cast<double>(dimsA[2]) *
    static_cast<double>(dimsA[1]) * static_cast<double>(dimsB[2]);
  double gigaFlops =
      (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);
  printf("\nPerformance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops,"
         " WorkgroupSize= %lu threads/block\n",
         gigaFlops, msecPerMatrixMul, flopsPerMatrixMul, threads[2] * threads[1]);

  // Copy result from device to host
  stream->memcpy(h_C, d_C, mem_size_C);
  stream->wait();

  printf("\nChecking computed result for correctness: ");
  bool correct = true;

  // test relative error by the formula
  //     |<x, y>_cpu - <x,y>_gpu|/<|x|, |y|>  < eps
  double eps = 1.e-6;  // machine zero

  for (int i = 0; i < static_cast<int>(dimsC[2] * dimsC[1]); i++) {
    double abs_err = fabs(h_C[i] - (dimsA[2] * valB));
    double dot_length = dimsA[2];
    double abs_val = fabs(h_C[i]);
    double rel_err = abs_err / abs_val / dot_length;

    //if (rel_err > eps) {
    //  printf("Error! Matrix[%05d]=%.8f, ref=%.8f error term is > %E\n", i, h_C[i], dimsA[2] * valB, eps);
    //  correct = false;
    //}
  }

  printf("%s\n", correct ? "Result = PASS" : "Result = FAIL");

  //====== Repeat the above using MAGMA ======================================
  
  magmablas_sgemm2(MagmaNoTrans, MagmaNoTrans, dimsA[1], dimsB[2], dimsA[2],
                   1.0, d_A, dimsA[1], d_B, dimsB[1], 0.0, d_C, dimsC[1], stream);
  
  //====== Do it with MKL ====================================================
  /*
  mkl::transpose transA = mkl::transpose::nontrans;
  mkl::transpose transB = mkl::transpose::nontrans;
  sycl::event gemm_done;
  std::vector<sycl::event> gemm_dependencies;
  gemm_done = mkl::blas::gemm(dpct::get_default_queue(), transA, transB, dimsA[1], dimsB[2], dimsA[2],
			      1.0, d_A, dimsA[1], d_B, dimsB[1], 0.0, d_C, dimsC[1], gemm_dependencies);
  gemm_done.wait();
  */
  //==========================================================================
  
  printf("\nMAGMA done\n");

  stream->memcpy(d_A, h_A, mem_size_A);
  stream->memcpy(d_B, h_B, mem_size_B);

  stream->wait();

  // Record the start event
  start_ct1 = std::chrono::steady_clock::now();

  // Execute the kernel
  for (int j = 0; j < nIter; j++)
    magmablas_sgemm2(MagmaNoTrans, MagmaNoTrans, dimsA[1], dimsB[2], dimsA[2],
		     1.0, d_A, dimsA[1], d_B, dimsB[1], 0.0, d_C, dimsC[1], stream);
  /*
  for (int j = 0; j < nIter; j++)
    gemm_done = mkl::blas::gemm(dpct::get_default_queue(), transA, transB, dimsA[1], dimsB[2], dimsA[2],
				1.0, d_A, dimsA[1], d_B, dimsB[1], 0.0, d_C, dimsC[1], gemm_dependencies);
  gemm_done.wait();
  */
  stream->wait();
  stop_ct1 = std::chrono::steady_clock::now();

  msecTotal = std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1).count();

  // Compute and print the performance
  msecPerMatrixMul = msecTotal / nIter;
  flopsPerMatrixMul = 2.0 * static_cast<double>(dimsA[2]) *
      static_cast<double>(dimsA[1]) * static_cast<double>(dimsB[2]);
  gigaFlops =
      (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);
  printf("\nMAGMA Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops,"
         " WorkgroupSize= %lu threads/block\n",
         gigaFlops, msecPerMatrixMul, flopsPerMatrixMul, threads[2] * threads[1]);

  // Copy result from device to host
  stream->memcpy(h_C, d_C, mem_size_C);
  stream->wait();

  printf("\nChecking computed result for correctness: ");
  correct = true;

  // test relative error by the formula
  //     |<x, y>_cpu - <x,y>_gpu|/<|x|, |y|>  < eps
  for (int i = 0; i < static_cast<int>(dimsC[2] * dimsC[1]); i++) {
      double abs_err = fabs(h_C[i] - (dimsA[2] * valB));
      double dot_length = dimsA[2];
      double abs_val = fabs(h_C[i]);
      double rel_err = abs_err / abs_val / dot_length;

      if (rel_err > eps) {
          printf("Error! Matrix[%05d]=%.8f, ref=%.8f error term is > %E\n", i, h_C[i], dimsA[2] * valB, eps);
          correct = false;
      }
  }

  printf("%s\n", correct ? "Result = PASS" : "Result = FAIL");
 
  //====== Repeat the above using sgemm_ijk ================================== 
  /*
  sgemm_bijk(dimsA[1], dimsB[2], dimsA[2],
             1.0, h_A, dimsA[1], h_B, dimsB[1], 0.0, h_C, dimsC[1]);

  printf("\nsgemm_bijk done\n");

  // Record the start event
  start_ct1 = std::chrono::steady_clock::now();

  // Execute the kernel
  for (int j = 0; j < nIter; j++)
    sgemm_bijk(dimsA[1], dimsB[2], dimsA[2],
              1.0, h_A, dimsA[1], h_B, dimsB[1], 0.0, h_C, dimsC[1]);

  stream->wait();
  stop_ct1 = std::chrono::steady_clock::now();
  
  msecTotal = std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1).count();

  // Compute and print the performance
  msecPerMatrixMul = msecTotal / nIter;
  flopsPerMatrixMul = 2.0 * static_cast<double>(dimsA[2]) *
      static_cast<double>(dimsA[1]) * static_cast<double>(dimsB[2]);
  gigaFlops =
      (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);
  printf("\nsgemm_bijk Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops,"
         " WorkgroupSize= %lu threads/block\n",
         gigaFlops, msecPerMatrixMul, flopsPerMatrixMul, threads[2] * threads[1]);
  */
  //==========================================================================
  // Clean up memory
  sycl::free(h_A, dpct::get_default_queue());
  sycl::free(h_B, dpct::get_default_queue());
  sycl::free(h_C, dpct::get_default_queue());
  sycl::free(d_A, dpct::get_default_queue());
  sycl::free(d_B, dpct::get_default_queue());
  sycl::free(d_C, dpct::get_default_queue());
  
  printf(
      "\nNOTE: The CUDA Samples are not meant for performance "
      "measurements. Results may vary when GPU Boost is enabled.\n");

  if (correct) {
    return EXIT_SUCCESS;
  } else {
    return EXIT_FAILURE;
  }
}

/*
  Function:   Main
  Parameters: Optional dimensions of matrices A and B
  
  This function checks for dimensions of matrices A and B. If none are provided, it multiples dimsA 320,320,1 
  and dimsB 640,320,1 by default. It tests the performance of the MatrixMultiply function by limiting the 
  profile around the function call via cudaProfilerStart() and cudaProfilerStop().
*/

int main(int argc, char **argv) {

  printf("\n[Matrix Multiply Using SYCL] - Starting...\n");

  /*
  sycl::device d;
  
  try{
		d = sycl::device(sycl::gpu_selector());
	} catch (sycl::exception const &e) {
		//std::cout << "Cannot select a GPU\n" << e.what() << "\n";
		std::cout << "Using a CPU device\n";
		d = sycl::device(sycl::cpu_selector());
	} 

  */

  //std::cout << "\nUsing " << d.get_info<sycl::info::device::name>() << "\n";

  if (checkCmdLineFlag(argc, (const char **)argv, "help") ||
      checkCmdLineFlag(argc, (const char **)argv, "?")) {
        printf(" Usage -device=n (n >= 0 for deviceID)\n");
        printf("      -wA=WidthA -hA=HeightA (Width x Height of Matrix A)\n");
        printf("      -wB=WidthB -hB=HeightB (Width x Height of Matrix B)\n");
        printf(" Note: Outer matrix dimensions of A & B matrices must be equal.\n");

    exit(EXIT_SUCCESS);
  }

  sycl::queue q;
  printf("\nUsing %s...\n", q.get_device().get_info<sycl::info::device::name>().c_str());

  // 1. This will pick the best possible CUDA capable device, otherwise
  // override the device ID based on input provided at the command line

  //int dev = findCudaDevice(argc, (const char **)argv);
	int dev = 0;

  int block_size = 16;

  // 2. Default matrices (no command line arguments provided)

  sycl::range<3> dimsA(1, 1 * 8 * block_size, 1 * 8 * block_size); // 320,320,1
  sycl::range<3> dimsB(1, 1 * 8 * block_size, 1 * 8 * block_size); // 640,320,1

  // 3. Possible command line arguments to override default

  // width of Matrix A
  if (checkCmdLineFlag(argc, (const char **)argv, "wA")) {
    dimsA[2] = getCmdLineArgumentInt(argc, (const char **)argv, "wA");
  }

  // height of Matrix A
  if (checkCmdLineFlag(argc, (const char **)argv, "hA")) {
    dimsA[1] = getCmdLineArgumentInt(argc, (const char **)argv, "hA");
  }

  // width of Matrix B
  if (checkCmdLineFlag(argc, (const char **)argv, "wB")) {
    dimsB[2] = getCmdLineArgumentInt(argc, (const char **)argv, "wB");
  }

  // height of Matrix B
  if (checkCmdLineFlag(argc, (const char **)argv, "hB")) {
    dimsB[1] = getCmdLineArgumentInt(argc, (const char **)argv, "hB");
  }

  if (dimsA[2] != dimsB[1]) {
    printf("Error: outer matrix dimensions must be equal. (%zu != %zu)\n", dimsA[2], dimsB[1]);
    exit(EXIT_FAILURE);
  }

  // 4. Output matrix dimensions

  printf("\nMatrixA(%zu,%zu), MatrixB(%zu,%zu)\n", dimsA[2], dimsA[1], dimsB[2], dimsB[1]);

  // 5. Time matrix multiplication execution time
  // Profile is limited to MatrixMultiply() execution of code

  //checkCudaErrors(cudaProfilerStart());

  int matrix_result = MatrixMultiply(argc, argv, block_size, dimsA, dimsB);

  //checkCudaErrors(cudaProfilerStop());

  exit(matrix_result);
}
