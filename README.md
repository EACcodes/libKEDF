libKEDF
======

A BSD3-licensed library for (advanced) KEDFs and their, hopefully advanced, implementation.

Currently implemented:
* von Weizsaecker
* Thomas-Fermi
* Wang-Teter (including other parametrizations)
* Taylor-expanded Wang-Govind-Carter in 1st and 2nd order
* Huang-Carter

with energy and gradient.

Supported technologies:
* OpenMP
* OpenCL

Required libraries:
* Armadillo
* BLAS/LAPACK compatible libraries (e. g., OpenBLAS, MKL, ...)
* an FFTW3-compliant FFT library (e. g., FFTW3, MKL, ...)
* clFFT (with OPENCL option)

Included technologies
* patched verion of rksuite (LGPL)
* libspline (LGPL)

Supported platforms:
* FreeBSD
* Linux

For basic compilation:
> mkdir build && cd build

> cmake ..

> make

which will create libKEDF.a, libspline.so, and librksuite.so. These must all be linked against from client code. Also, there is KEDFClient, a bare-bones client implementation.

For advanced compilation:
> mkdir build && cd build

> cmake .. -DOPENMP=ON -DOPENCL=ON -DMPI=ON -DCLFFT_ROOT_DIR=/path/to/clFFT/root -DFFTW3_ROOT_DIR=/path/to/fftw3/root -DARMADILLO_INCLUDE_DIR=/path/to/armadillo/include -DARMADILLO_LIBRARY=/path/to/arma/library.so

NOTE: All or some of the options can be used.

> make

