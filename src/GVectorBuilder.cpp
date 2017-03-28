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
#include <complex>
#include <math.h>
#include "GVectorBuilder.hpp"
using namespace arma;

void GVectorBuilder::buildGVectors(const shared_ptr<mat> cellVectors,
        const size_t recX, const size_t recY, const size_t recZ,
        shared_ptr<cube>& gNorms, shared_ptr<cube>& gVectorsX, shared_ptr<cube>& gVectorsY, 
        shared_ptr<cube>& gVectorsZ){
    
    buildGVectors(cellVectors, recX, recY, recZ, 0, 0, 0, recX, recY, recZ, gNorms,
            gVectorsX, gVectorsY, gVectorsZ);
}

void GVectorBuilder::buildGVectors(const shared_ptr<mat> cellVectors,
        const size_t recX, const size_t recY, const size_t recZ,
        shared_ptr<fcube>& gNorms, shared_ptr<fcube>& gVectorsX, shared_ptr<fcube>& gVectorsY,
        shared_ptr<fcube>& gVectorsZ){
    
    buildGVectors(cellVectors, recX, recY, recZ, 0, 0, 0, recX, recY, recZ, gNorms,
            gVectorsX, gVectorsY, gVectorsZ);
}

void GVectorBuilder::buildGVectors(const shared_ptr<mat> cellVectors,
            const size_t recX, const size_t recY, const size_t recZ,
            const size_t offX, const size_t offY, const size_t offZ,
            const size_t endX, const size_t endY, const size_t endZ,
            shared_ptr<cube> & gNorms, shared_ptr<cube>& gVectorsX,
            shared_ptr<cube>& gVectorsY, shared_ptr<cube>& gVectorsZ){
 
    if(DEBUG){
        cout << "Computing g vectors and norms for the following cell definition:" << endl;
        cellVectors->print("\t cell vectors:");
        cout << "\t reciprocal grid size: " << recX << "\t" << recY << "\t" << recZ << endl;
        cout << "\t reciprocal grid offset: " << offX << "\t" << offY << "\t" << offZ << endl;
        cout << "\t reciprocal grid end: " << endX << "\t" << endY << "\t" << endZ << endl;
    }
    
    const double twoPi = 2.0* M_PI;
    
    // calculate the reciprocal cell vectors
    mat::fixed<3,3> cellReciprocal = cellVectors.get()->t().i().t();
    cellReciprocal *= twoPi;

    vec::fixed<3> mVector;

    #pragma omp parallel for default(none) shared(gNorms, gVectorsX, gVectorsY, gVectorsZ, cellReciprocal) private(mVector)
    for(size_t k = offZ; k < endZ; ++k){
        mVector.at(2) = (k <= recZ/2) ? k : double(k)-double(recZ);
        for(size_t j = offY; j < endY; ++j){
            mVector.at(1) = (j <= recY/2) ? j : double(j)-double(recY);;
            for(size_t i = offX; i < endX; ++i){
                mVector.at(0) = i;
                
                const vec::fixed<3> gPoint = cellReciprocal * mVector;

                gVectorsX->at(i-offX,j-offY,k-offZ) = gPoint.at(0);
                gVectorsY->at(i-offX,j-offY,k-offZ) = gPoint.at(1);
                gVectorsZ->at(i-offX,j-offY,k-offZ) = gPoint.at(2);

#ifdef LIBKEDF_SUPERDETAIL_GVECTORS
                cout << "for " << k << " "<< j << " " << i << endl;
                cout << "gvector X " << gVectorsX->at(i-offX,j-offY,k-offZ) << endl;
                cout << "gvector Y " << gVectorsY->at(i-offX,j-offY,k-offZ) << endl;
                cout << "gvector Z " << gVectorsZ->at(i-offX,j-offY,k-offZ) << endl;
#endif
                gNorms->at(i-offX,j-offY,k-offZ) = norm(gPoint);
            }
        }
    }
        
    if(DEBUG){
        gNorms->print("g-Norms");
        gVectorsX->print("g-Vectors (x-component)");
        gVectorsY->print("g-Vectors (y-component)");
        gVectorsZ->print("g-Vectors (z-component)");
    }
}

void GVectorBuilder::buildGVectors(const shared_ptr<mat> cellVectors,
            const size_t recX, const size_t recY, const size_t recZ,
            const size_t offX, const size_t offY, const size_t offZ,
            const size_t endX, const size_t endY, const size_t endZ,
            shared_ptr<fcube>& gNorms, shared_ptr<fcube>& gVectorsX,
            shared_ptr<fcube>& gVectorsY, shared_ptr<fcube>& gVectorsZ){
 
    if(DEBUG){
        cout << "Computing float g vectors and norms for the following cell definition:" << endl;
        cellVectors->print("\t cell vectors:");
        cout << "\t reciprocal grid size: " << recX << "\t" << recY << "\t" << recZ << endl;
        cout << "\t reciprocal grid offset: " << offX << "\t" << offY << "\t" << offZ << endl;
        cout << "\t reciprocal grid end: " << endX << "\t" << endY << "\t" << endZ << endl;
    }
    
    const double twoPi = 2.0* M_PI;
    
    // calculate the reciprocal cell vectors
    mat::fixed<3,3> cellReciprocal = cellVectors.get()->t().i().t();
    cellReciprocal *= twoPi;

    vec::fixed<3> mVector;

    #pragma omp parallel for default(none) shared(gNorms, gVectorsX, gVectorsY, gVectorsZ, cellReciprocal) private(mVector)
    for(size_t k = offZ; k < endZ; ++k){
        mVector.at(2) = (k <= recZ/2) ? k : double(k)-double(recZ);
        for(size_t j = offY; j < endY; ++j){
            mVector.at(1) = (j <= recY/2) ? j : double(j)-double(recY);;
            for(size_t i = offX; i < endX; ++i){
                mVector.at(0) = i;
                
                const vec::fixed<3> gPoint = cellReciprocal * mVector;

                gVectorsX->at(i-offX,j-offY,k-offZ) = (float) gPoint.at(0);
                gVectorsY->at(i-offX,j-offY,k-offZ) = (float) gPoint.at(1);
                gVectorsZ->at(i-offX,j-offY,k-offZ) = (float) gPoint.at(2);

#ifdef LIBKEDF_SUPERDETAIL_GVECTORS
                cout << "for " << k << " "<< j << " " << i << endl;
                cout << "gvector X " << gVectorsX->at(i-offX,j-offY,k-offZ) << endl;
                cout << "gvector Y " << gVectorsY->at(i-offX,j-offY,k-offZ) << endl;
                cout << "gvector Z " << gVectorsZ->at(i-offX,j-offY,k-offZ) << endl;
#endif
                gNorms->at(i-offX,j-offY,k-offZ) = (float) norm(gPoint);
            }
        }
    }
        
    if(DEBUG){
        gNorms->print("float g-Norms");
        gVectorsX->print("float g-Vectors (x-component)");
        gVectorsY->print("float g-Vectors (y-component)");
        gVectorsZ->print("float g-Vectors (z-component)");
    }
}
