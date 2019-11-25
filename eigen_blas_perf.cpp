#include <iostream>
#include <Eigen/Dense>
#include <malloc.h>
#include <cblas.h>
#include <time.h>
#include <x86intrin.h>

using namespace Eigen;

#define BILLION  1E9;

#define M 1024
#define N 1024
#define K 1024

#define LOOP 640 
#define TIME(func, desc) {\
		printf("%s\n", desc); \
		struct timespec t1, t2; \
                clock_gettime(CLOCK_MONOTONIC, &t1);	\
		func;\
                clock_gettime(CLOCK_MONOTONIC, &t2);	\
	        double accum = double(t2.tv_sec - t1.tv_sec) + ( t2.tv_nsec - t1.tv_nsec ) / BILLION; \
		printf("\ttime taken: %.2lf second.\n", \
					accum); \
}
void int_perf(){

    // multiplication by eigen
    MatrixXi a = (MatrixXd::Random(M, K) * 1024).cast<int>();
    MatrixXi b = (MatrixXd::Random(K, N) * 1024).cast<int>();
    MatrixXi d(M, N);
    TIME(for (int i = 0; i < LOOP; i++) {
    	d = a*b;
    	d(0, 0) = i;}, "eigen integer")
    std::cout << "\t[" << d(0, 0) << " " << d(0, 1) << " ...]" << std::endl;

    // multiplication by openblas
    const float alpha=1;
    const float beta=0;
    float *a_arr = (float *)memalign(64, M*K*sizeof(float));
    float *b_arr = (float *)memalign(64, K*N*sizeof(float));
    float *c_arr = (float *)memalign(64, M*N*sizeof(float));

    for (int i = 0; i < M; i++) {
       for (int j = 0; j < K; j++) {
           a_arr[i * K + j] = a(i, j);
       }
    }

    for (int i = 0; i < K; i++) {
       for (int j = 0; j < N; j++) {
           b_arr[i * N + j] = b(i, j);
       }
    }
    memset(c_arr, 0, M*N*sizeof(float));
    TIME(for (int i = 0; i < LOOP; i++) {
		cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, a_arr, K, b_arr, N, beta, c_arr, N);
		c_arr[0] = i;},
		"blas integer")
    std::cout <<"\t[" << c_arr[0] << " " << c_arr[1] << " ...]" << std::endl;

    free(a_arr);
    free(b_arr);
    free(c_arr);
}

void double_perf(){
    MatrixXd a = MatrixXd::Random(M, K);
    MatrixXd b = MatrixXd::Random(K, N);
    MatrixXd d(M, N);

    TIME(for (int i = 0; i < LOOP; i++) {
		d = a * b;
		d(0, 0) = i;},
		"eigen double")
    std::cout << "\t[" << d(0, 0) << " " << d(0, 1) << " ...]" << std::endl;

    // multiplication by openblas
    const float alpha=1;
    const float beta=0;
    double *a_arr = (double *)memalign(64, M*K*sizeof(double));
    double *b_arr = (double *)memalign(64, K*N*sizeof(double));
    double *c_arr = (double *)memalign(64, M*N*sizeof(double));

    for (int i = 0; i < M; i++) {
       for (int j = 0; j < K; j++) {
           a_arr[i * K + j] = a(i, j);
       }
    }

    for (int i = 0; i < K; i++) {
       for (int j = 0; j < N; j++) {
           b_arr[i * N + j] = b(i, j);
       }
    }
    TIME(
	for (int i = 0; i < LOOP; i++) {
    		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, a_arr, K, b_arr, N, beta, c_arr, N);
		c_arr[0] = i;},
		"blas double")
    std::cout <<"\t[" << c_arr[0] << " " << c_arr[1] << " ...]" << std::endl;

    free(a_arr);
    free(b_arr);
    free(c_arr);
}

int main()
{
    int_perf();
    double_perf();
    return 0;
}

