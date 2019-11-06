#include <iostream>
#include <Eigen/Dense>
#include <malloc.h>
#include <cblas.h>
#include <x86intrin.h>

using namespace Eigen;

#define M 4000
#define N 2000
#define K 1024

#define TIME(func, desc) {\
		printf("%s\n", desc); \
		clock_t t1 = clock(); \
		func;\
		clock_t t2 = clock(); \
		printf("\ttime taken: %.2f second.\n", \
					((float)t2 - (float)t1) / CLOCKS_PER_SEC); \
}
void int_perf(){

    // multiplication by eigen
    MatrixXi a = (MatrixXd::Random(M, K) * 1024).cast<int>();
    MatrixXi b = (MatrixXd::Random(K, N) * 1024).cast<int>();
    MatrixXi d(M, N);
    TIME((d=a*b), "eigen integer")
    std::cout << "\t[" << d(0, 0) << " ...]" << std::endl;

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

    TIME(cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, a_arr, K, b_arr, N, beta, c_arr, N), 
		    "blas integer")
    std::cout <<"\t[" << c_arr[0] << " ...]" << std::endl;

    free(a_arr);
    free(b_arr);
    free(c_arr);
}

void double_perf(){
    MatrixXd a = MatrixXd::Random(M, K);
    MatrixXd b = MatrixXd::Random(K, N);
    MatrixXd d(M, N);

    TIME((d = a * b), "eigen double")
    std::cout << "\t[" << d(0, 0) << " ...]" << std::endl;

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

    TIME(cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, a_arr, K, b_arr, N, beta, c_arr, N),
		    "blas double")
    std::cout <<"\t[" << c_arr[0] << " ...]" << std::endl;

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

