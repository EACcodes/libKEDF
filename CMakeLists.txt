cmake_minimum_required(VERSION 3.1)
project(libKEDF C CXX)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake-modules)

option(OPENMP "Use OpenMP for, e.g., the FFT backend." OFF)
option(OPENCL "Use OpenCL for, e.g., the FFT backend (powered by clFFT)." ON)
option(JAVA "Compile with Java interface support. Requires JNI." ON)
option(HAS_PRINTF "OpenCL runtime exposes printf() function as per OCL standard 1.2." ON)
option(HAS_POWR "OpenCL runtime exposes powr() function as per OCL standard 1.2." ON)

if(HAS_PRINTF)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DLIBKEDF_HAS_PRINTF=1}")
endif()

if(HAS_POWR)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DLIBKEDF_HAS_POWR=1}")
endif()

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'RelWithDebInfo specified.")
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "RelWithDebInfo")
endif()

add_subdirectory(src)
add_subdirectory(client)
#add_subdirectory(tests)
add_subdirectory(rksuite)
add_subdirectory(spline)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
# add the rksuite  header
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/rksuite)
# add the spline header
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/spline)
# for some client-internal headers
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/client)

if (JAVA)
    find_package(JNI REQUIRED)
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/java/src/c)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/java/src/c)
endif()

if (JAVA)
    include_directories(${JNI_INCLUDE_DIRS})
endif()

# find the correct FFTW3 libraries to link against based on the parallelization chosen
if (OPENMP)
  find_package(OpenMP REQUIRED)
  find_package(FFTW3 REQUIRED COMPONENTS double openmp)
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(LIBKEDF_GRIDS ${LIBKEDF_SHMEM_SOURCE})
else ()
  find_package(FFTW3 REQUIRED COMPONENTS double)
  set(LIBKEDF_GRIDS ${LIBKEDF_SHMEM_SOURCE})
endif()

include_directories(${FFTW3_INCLUDE_DIRS})

# find armadillo
find_package(Armadillo REQUIRED)
include_directories(${ARMADILLO_INCLUDE_DIRS})

# find BLAS/LAPACK for armadillo
if (${CMAKE_CXX_COMPILER_ID} STREQUAL "Intel")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mkl=sequential")
else()
  find_package(BLAS REQUIRED)
  find_package(LAPACK REQUIRED)
endif()

# we need this on FreeBSD since OpenBLAS requires gfortran
if(${CMAKE_SYSTEM_NAME} MATCHES "FreeBSD")
  set( CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} -Wl,-rpath=/usr/local/lib/gcc49")
endif(${CMAKE_SYSTEM_NAME} MATCHES "FreeBSD")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Werror -Wno-comment -Wunused-value -std=c++14 -fPIC")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Werror -Wunused-value")

if (${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU" OR ${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations")
  if(NOT OPENMP)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unknown-pragmas")
  endif()
else (${CMAKE_CXX_COMPILER_ID} STREQUAL "Intel" AND NOT OPENMP)
  # unset openmp pragma warning
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -diag-disable 3180")
endif()

# wide alignment required. Comment if alignment is not supported, change value if 64byte is too wide.
add_definitions("-DLIBKEDF_ALIGNMENT=64")

# to get more details (verbose!) uncomment this
#add_definitions("-DLIBKEDF_DEBUG=1")

# Make sure we use O0 for debug and add preprocessor definition
if (${CMAKE_BUILD_TYPE} STREQUAL "Debug")
  add_definitions("-DLIBKEDF_DEBUG=1")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0")
endif()

if (${CMAKE_BUILD_TYPE} STREQUAL "Release" OR ${CMAKE_BUILD_TYPE} STREQUAL "RelWithDebInfo")
  if (${CMAKE_CXX_COMPILER_ID} STREQUAL "Intel")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -xHost")
  else ()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")
#    XXX under clang 3.8 as bundled with FreeBSD CURRENT ATM [20160323] this does NOT work
#    if(${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
#        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -flto")
#        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -flto")
#        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -flto") # XXX: MODULE_LINKER_FLAGS?
#    endif()
  endif()
endif()

if (OPENCL)
  add_definitions("-DLIBKEDF_OCL=1")
  find_package(OpenCL REQUIRED)
  find_package(clFFT REQUIRED)
  include_directories(${OpenCL_INCLUDE_DIRS})
  include_directories(${CLFFT_INCLUDE_DIRS})
  set(LIBKEDF_GRIDS ${LIBKEDF_OCL_SOURCE} ${LIBKEDF_SHMEM_SOURCE})
endif()

# thanks to the decision of making a public domain software (rksuite.f) viral (rksuite.cpp)
# this stupidity is necessary to be license compliant with the LGPL and keeping BSD for OUR code.
# why do we bundle the rksuite code? because the upstream code (i.e., the translation that was worth re-licensing)
# does not compile actually w/ C++11 and w/o warnings. changes are under LGPL.
#
# of course, people can feel free to, in accordance w/ the LGPL, swap this out
# for "their" implementation/compilation of rksuite.
add_library(rksuite SHARED ${RKSUITE_SOURCE})

add_library(spline SHARED ${SPLINE_SOURCE})

# use below to force STATIC library building. above is "environment dependent" which typically means shared
#add_library(KEDF STATIC ${LIBKEDF_SOURCE} ${LIBKEDF_GRIDS})
# to force a shared library, use this:
#add_library(KEDF SHARED ${LIBKEDF_SOURCE} ${LIBKEDF_GRIDS})
add_library(KEDF ${LIBKEDF_SOURCE} ${LIBKEDF_GRIDS})

add_executable(KEDFClient ${CLIENT_SOURCE})

if (JAVA)
    # this needs to be shared for JNI!
    add_library(KEDFjni SHARED ${LIBKEDF_JNI_SOURCE})
    target_link_libraries(KEDFjni KEDF)
    target_link_libraries(KEDFjni ${JNI_LIBRARIES})
endif()

target_link_libraries(KEDF rksuite spline)
target_link_libraries(KEDF ${FFTW3_LIBRARIES})

target_link_libraries(KEDFClient KEDF)

target_link_libraries(KEDF ${ARMADILLO_LIBRARIES})
target_link_libraries(KEDFClient ${ARMADILLO_LIBRARIES})

if (${CMAKE_CXX_COMPILER_ID} STREQUAL "Intel")
else()
  target_link_libraries(KEDF ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
endif()

if (OPENCL)
  target_link_libraries(KEDF ${CLFFT_LIBRARIES})
  target_link_libraries(KEDF ${OpenCL_LIBRARIES})
endif()

# we need this on FreeBSD since OpenBLAS requires gfortran
if(${CMAKE_SYSTEM_NAME} MATCHES "FreeBSD")
  LINK_DIRECTORIES( ${LINK_DIRECTORIES} /usr/local/lib/gcc49)
  target_link_libraries(KEDF /usr/local/lib/gcc49/libgcc_s.so)
  target_link_libraries(KEDF /usr/local/lib/gcc49/libgfortran.so)
endif(${CMAKE_SYSTEM_NAME} MATCHES "FreeBSD")

set( PROJECT_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include )
install(DIRECTORY ${PROJECT_INCLUDE_DIR}/
  DESTINATION "include"
  FILES_MATCHING PATTERN "*.h"
  PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ)

install(DIRECTORY ${PROJECT_INCLUDE_DIR}/
  DESTINATION "include"
  FILES_MATCHING PATTERN "*.hpp"
  PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ)

if(JAVA)
  INSTALL(TARGETS KEDFClient KEDF KEDFjni spline rksuite
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
  )
else()
  INSTALL(TARGETS KEDFClient KEDF spline rksuite
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
  )
endif()
