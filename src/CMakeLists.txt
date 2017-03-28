set(LIBKEDF_SOURCE
src/FermiWaveVector.cpp
src/GridFactory.cpp
src/GVectorBuilder.cpp
src/HelperFunctions.cpp
src/HuangCarterKernel.cpp
src/IntKernelODE.cpp
src/MemoryFunctions.cpp
src/StressTensor.cpp
src/WangGovindCarterKernel.cpp
src/WangTeterKernel.cpp
src/libKEDF.cpp
PARENT_SCOPE)

set(LIBKEDF_SHMEM_SOURCE
src/CartesianOOPGrid.cpp
PARENT_SCOPE)

set(LIBKEDF_OCL_SOURCE
src/CartesianOCLOOPGrid.cpp
PARENT_SCOPE)
