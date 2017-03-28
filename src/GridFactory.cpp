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
#ifdef _OPENMP
#include <fftw3.h>
#endif
#include "GridFactory.hpp"
#include "BasicGridComputer.hpp"
#include "CartesianOOPGrid.hpp"
#ifdef LIBKEDF_OCL
#include <clFFT.h>
#include "CartesianOCLOOPGrid.hpp"
//   # include "OCLGridComputer.hpp"
#endif
using namespace arma;

FourierGrid* GridFactory::constructFourierGrid(const size_t globX, const size_t globY,
            const size_t globZ, const shared_ptr<mat> cellVectors, const string config){
    
#ifdef LIBKEDF_DEBUG
    cout << "Config for grid factory is " << config << endl;
#endif
    
#ifdef _OPENMP
    fftw_init_threads();
#endif

    if(config.compare("fftw3,out-of-place") == 0){
        return (new CartesianOOPGrid(globX,globY,globZ, cellVectors));
#ifdef LIBKEDF_OCL
    } else if(config.compare("clfft,out-of-place") == 0){
        
        size_t platformNo = 0;
        size_t deviceNo = 4;
        
        clfftSetupData fftSetup;
        cl_int err;
        err = clfftInitSetupData(&fftSetup);
        if(err != CL_SUCCESS){
            cerr << "ERROR in setting up setup data for clFFT API setup: " << err << endl;
            throw runtime_error("ERROR in setting up setup data for clFFT API setup.");
        }
        err = clfftSetup(&fftSetup);
        if(err != CL_SUCCESS){
            cerr << "ERROR in clFFT API setup: " << err << endl;
            throw runtime_error("ERROR in clFFT API setup.");
        }
        
        CartesianOCLOOPGrid* grid = (new CartesianOCLOOPGrid(globX,globY,globZ, cellVectors, platformNo, deviceNo));
        
        return grid;
#endif
    } else {
        cerr << "Unknown FourierGrid definition " << config << endl;
        throw runtime_error("Unknown FourierGrid definition " + config);
    }
}
