# mathlib_perf
Demo the performance of math library - eigen, openblas, intel MKL and numpy

# Install MKL
## Download MKL and unpack
```
tar zxvf l_mkl_2019.5.281.tgz
cd l_mkl_2019.5.281
./install.sh
source /opt/intel/bin/compilervars.sh intel64
```

## compile
```
gcc -I /opt/intel/mkl/include/ SOURCE.c -lmkl_rt -L/opt/intel/mkl/lib/intel64 -L/opt/intel/lib/intel64
```
