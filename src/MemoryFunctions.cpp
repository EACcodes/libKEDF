/* 
 * Copyright (c) 2015-2016, Princeton University, Johannes M Dieterich, Emily A Carter
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation and/or
 * other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its contributors may
 * be used to endorse or promote products derived from this software without specific
 * prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
 * INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
 * NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */
#include "HelperFunctions.hpp"
using namespace arma;

unique_ptr<cube, void (*) (cube*)> MemoryFunctions::allocateScratch(const size_t rows, const size_t cols, const size_t slices){
    
#ifdef LIBKEDF_NO_ALIGNED_MEMORY
    unique_ptr<cube, void (*) (cube*)> ptr (new cube(rows,cols,slices),
            [](cube* cube){
                delete cube;
            });
            
    return ptr;
#else
    const size_t alignment = LIBKEDF_ALIGNMENT;
    const size_t totalDims = rows*cols*slices*sizeof(double);
    double* rawMem;
    const int error = posix_memalign((void**)&rawMem,alignment,totalDims);
    if(error != 0){
        cerr << "ERROR: Can't get aligned memory: " << error << ", size: " << totalDims << ", alignment: " << alignment << endl;
        throw runtime_error("No aligned memory could be allocated.");
    }
    
    unique_ptr<cube, void (*) (cube*)> ptr (new cube(rawMem,rows,cols,slices,false,true),
            [](cube* cube){
                free(cube->memptr());
                delete cube;
            });
    
    return ptr;
#endif    
}

unique_ptr<fcube, void (*) (fcube*)> MemoryFunctions::allocateScratchFloat(const size_t rows, const size_t cols, const size_t slices){
    
#ifdef LIBKEDF_NO_ALIGNED_MEMORY
    unique_ptr<cube, void (*) (cube*)> ptr (new cube(rows,cols,slices),
            [](cube* cube){
                delete cube;
            });
            
    return ptr;
#else
    const size_t alignment = LIBKEDF_ALIGNMENT;
    const size_t totalDims = rows*cols*slices*sizeof(float);
    float* rawMem;
    const int error = posix_memalign((void**)&rawMem,alignment,totalDims);
    if(error != 0){
        cerr << "ERROR: Can't get aligned memory: " << error << ", size: " << totalDims << ", alignment: " << alignment << endl;
        throw runtime_error("No aligned memory could be allocated.");
    }
    
    unique_ptr<fcube, void (*) (fcube*)> ptr (new fcube(rawMem,rows,cols,slices,false,true),
            [](fcube* cube){
                free(cube->memptr());
                delete cube;
            });
    
    return ptr;
#endif    
}

unique_ptr<vec, void (*) (vec*)> MemoryFunctions::allocateScratch(const size_t elements){

#ifdef LIBKEDF_NO_ALIGNED_MEMORY
    unique_ptr<cube, void (*) (vec*)> ptr (new vec(elements),
            [](vec* vec){
                delete vec;
            });
            
    return ptr;
#else
    const size_t alignment = LIBKEDF_ALIGNMENT;
    const size_t totalDims = elements*sizeof(double);
    double* rawMem;
    const int error = posix_memalign((void**)&rawMem,alignment,totalDims);
    if(error != 0){
        cerr << "ERROR: Can't get aligned memory: " << error << ", size: " << totalDims << ", alignment: " << alignment << endl;
        throw runtime_error("No aligned memory could be allocated.");
    }
    
    unique_ptr<vec, void (*) (vec*)> ptr (new vec(rawMem,elements,false,true),
            [](vec* vec){
                free(vec->memptr());
                delete vec;
            });
    
    return ptr;
#endif
}

unique_ptr<cx_cube, void (*) (cx_cube*)> MemoryFunctions::allocateReciprocalScratch(const size_t rows, const size_t cols, const size_t slices){
    
#ifdef LIBKEDF_NO_ALIGNED_MEMORY
    unique_ptr<cx_cube, void (*) (cx_cube*)> ptr (new cx_cube(rows,cols,slices),
            [](cx_cube* cube){
                delete cube;
            });
            
    return ptr;
#else
    const size_t alignment = LIBKEDF_ALIGNMENT;
    const size_t totalDims = rows*cols*slices*sizeof(complex<double>);
    complex<double>* rawMem;
    const int error = posix_memalign((void**)&rawMem,alignment,totalDims);
    if(error != 0){
        cerr << "ERROR: Can't get aligned memory: " << error << ", size: " << totalDims << ", alignment: " << alignment << endl;
        throw runtime_error("No aligned memory could be allocated.");
    }
    
    unique_ptr<cx_cube, void (*) (cx_cube*)> ptr (new cx_cube(rawMem,rows,cols,slices,false,true),
            [](cx_cube* cube){
                free(cube->memptr());
                delete cube;
            });
            
    return ptr;
#endif
}
    
cube* MemoryFunctions::allocateScratchCube(const size_t rows, const size_t cols, const size_t slices){
#ifdef LIBKEDF_NO_ALIGNED_MEMORY
    return new cube(rows,cols,slices);
#else
    const size_t alignment = LIBKEDF_ALIGNMENT;
    const size_t totalDims = rows*cols*slices*sizeof(double);
    double* rawMem;
    const int error = posix_memalign((void**)&rawMem,alignment,totalDims);
    if(error != 0){
        cerr << "ERROR: Can't get aligned memory: " << error << ", size: " << totalDims << ", alignment: " << alignment << endl;
        throw runtime_error("No aligned memory could be allocated.");
    }
    
    return new cube(rawMem,rows,cols,slices,false,true);
#endif
}

fcube* MemoryFunctions::allocateScratchCubeFloat(const size_t rows, const size_t cols, const size_t slices){
#ifdef LIBKEDF_NO_ALIGNED_MEMORY
    return new fcube(rows,cols,slices);
#else
    const size_t alignment = LIBKEDF_ALIGNMENT;
    const size_t totalDims = rows*cols*slices*sizeof(float);
    float* rawMem;
    const int error = posix_memalign((void**)&rawMem,alignment,totalDims);
    if(error != 0){
        cerr << "ERROR: Can't get aligned memory: " << error << ", size: " << totalDims << ", alignment: " << alignment << endl;
        throw runtime_error("No aligned memory could be allocated.");
    }
    
    return new fcube(rawMem,rows,cols,slices,false,true);
#endif
}
cx_cube* MemoryFunctions::allocateReciprocalScratchCube(const size_t rows, const size_t cols, const size_t slices){
#ifdef LIBKEDF_NO_ALIGNED_MEMORY
    return new cx_cube(rows,cols,slices);
#else
    const size_t alignment = LIBKEDF_ALIGNMENT;
    const size_t totalDims = rows*cols*slices*sizeof(complex<double>);
    complex<double>* rawMem;
    const int error = posix_memalign((void**)&rawMem,alignment,totalDims);
    if(error != 0){
        cerr << "ERROR: Can't get aligned memory: " << error << ", size: " << totalDims << ", alignment: " << alignment << endl;
        throw runtime_error("No aligned memory could be allocated.");
    }
    
    return new cx_cube(rawMem,rows,cols,slices,false,true);
#endif
}
