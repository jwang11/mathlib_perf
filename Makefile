all: eigen_openblas_perf eigen_mkl_perf

eigen_openblas_perf: eigen_blas_perf.cpp
	g++ -O3 -o $@ $^ -I/usr/include/eigen3 -lopenblas
       
eigen_mkl_perf: eigen_blas_perf.cpp
	g++ -O3 -o $@ $^ -I/usr/include/eigen3 -lmkl_rt -L/opt/intel/mkl/lib/intel64 -L/opt/intel/lib/intel64

.PHONY: clean
clean:
	@rm -f eigen_openblas_perf eigen_mkl_perf
