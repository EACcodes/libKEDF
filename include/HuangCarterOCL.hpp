/* 
 * Copyright (c) 2016, Princeton University, Johannes M Dieterich, Emily A Carter
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

/* 
 * Author: Johannes M Dieterich
 */

#ifndef HUANGCARTEROCL_HPP
#define HUANGCARTEROCL_HPP

#include <CL/cl.h>
#include <string>
#include "CartesianOCLOOPGrid.hpp"
using namespace arma;
using namespace std;

template<>
class HuangCarter<CartesianOCLOOPGrid>: public KEDF<CartesianOCLOOPGrid> {
    
public:
    
    HuangCarter(CartesianOCLOOPGrid* example, const size_t numEta, const double etaStep, unique_ptr<vec> w, unique_ptr<vec> w1,
            const double alpha, const double beta, const double lambda,
            const double rhoS, const double c, const double refRatio,
            const double cutoffDens, const bool trashEmptyBins, const size_t cutoffEmptyBin)
    : _numEta(numEta), _etaStep(etaStep), _w(move(w)), _w1(move(w1)), _alpha(alpha), _beta(beta), _hc_lambda(lambda),
            _rhoS(rhoS), _c(c), _midXi(cbrt(3.0*M_PI*M_PI*_rhoS)), _refRatio(refRatio),
            _cutoffDens(cutoffDens) {
        
        // split the w/w' tables up into two: one more used, smaller one and one less used, larger one
        const size_t highSize = _numEta - _KERNELSPLIT + 1; // +1 to have some overlap in the case that ind == KERNELSPLIT-1, ind+1 == KERNELSPLIT
        _wLow = make_unique<vec>(_KERNELSPLIT);
        _wHigh = make_unique<vec>(highSize);
        _w1Low = make_unique<vec>(_KERNELSPLIT);
        _w1High = make_unique<vec>(highSize);
        
        const size_t lowLength = _KERNELSPLIT*sizeof(double);
        const size_t highLength = highSize*sizeof(double);
        
        double* wP = _w->memptr();
        double* w1P = _w1->memptr();
        
        double* wLP = _wLow->memptr();
        double* wHP = _wHigh->memptr();
        double* w1LP = _w1Low->memptr();
        double* w1HP = _w1High->memptr();
        
        memcpy(wLP, wP, lowLength);
        memcpy(wHP, &wP[_KERNELSPLIT-1], highLength);
        memcpy(w1LP, w1P, lowLength);
        memcpy(w1HP, &w1P[_KERNELSPLIT-1], highLength);
        
        setupKernels(example);
        
        const size_t wSize = _numEta*sizeof(double);
        
        cl_int err;
        w0Buff = clCreateBuffer(_ctx, CL_MEM_READ_ONLY, wSize, NULL, &err);
        if(err != CL_SUCCESS){
            cerr << "ERROR in allocating GPU memory of size " << wSize << " with error " << err << endl;
            throw runtime_error("Failed to allocate GPU memory");
        }
        w1Buff = clCreateBuffer(_ctx, CL_MEM_READ_ONLY, wSize, NULL, &err);
        if(err != CL_SUCCESS){
            cerr << "ERROR in allocating GPU memory of size " << wSize << " with error " << err << endl;
            throw runtime_error("Failed to allocate GPU memory");
        }
        err = clEnqueueWriteBuffer(_queue, w0Buff, CL_FALSE, 0, wSize, _w->memptr(), 0, NULL, NULL );
        if(err != CL_SUCCESS){
            cerr << "ERROR to write w0 data to GPU " << err << endl;
            throw runtime_error("Failed to write w0 data to GPU.");
        }
        err = clEnqueueWriteBuffer(_queue, w1Buff, CL_FALSE, 0, wSize, _w1->memptr(), 0, NULL, NULL );
        if(err != CL_SUCCESS){
            cerr << "ERROR to write w1 data to GPU " << err << endl;
            throw runtime_error("Failed to write w1 data to GPU.");
        }
    }
    
    ~HuangCarter(){
        clReleaseCommandQueue(_queue);
        clReleaseContext(_ctx);
        clReleaseProgram(hcOCLProg);
        clReleaseKernel(hcE1Kernel);
        clReleaseKernel(hcE2Kernel);
        clReleaseKernel(hcE3Kernel);
        clReleaseKernel(hcE3ZeroKernel);
        clReleaseKernel(hcE4Kernel);
        clReleaseKernel(hcP0Kernel);
        clReleaseKernel(hcP1Kernel);
        clReleaseKernel(hcP2Kernel);
        clReleaseKernel(hcP2ZeroKernel);
        clReleaseKernel(hcP3Kernel);
        clReleaseKernel(hcP4Kernel);
        clReleaseKernel(hcP5Kernel);
        clReleaseKernel(hcP6Kernel);
        clReleaseKernel(hcP7Kernel);
        clReleaseKernel(binKernel);
        clReleaseKernel(calcXiKernel);
        clReleaseKernel(calcXiAndSSKernel);
        clReleaseKernel(hc10SimpleKernel);
        clReleaseMemObject(w0Buff);
        clReleaseMemObject(w1Buff);
    }
    
    string getMethodDescription() const {
        return "Huang-Carter (2010) KEDF (OpenCL version)";
    }
    
    vector<string> getCitations() const {
        vector<string> citations(0);
        citations.push_back("sorry, not yet");
    
        return citations;
    }

    vector<string> getWorkingEquations() const {
        
        vector<string> citations(0);
        citations.push_back("C. Huang and E. A. Carter, Phys. Rev. B 81, 045206 (2010).");

        return citations;
    }
    

    double calcEnergy(const CartesianOCLOOPGrid& grid) const {
        
        
        unique_ptr<CartesianOCLOOPGrid> workGrid = grid.duplicate();
        workGrid->transferRealToGPU();
        cl_mem densBuff = workGrid->getRealGPUBuffer();
        
        unique_ptr<CartesianOCLOOPGrid> workGridA = grid.duplicate();
        
        const size_t nSlices = workGrid->getGridPointsZ();
        const size_t nRows = workGrid->getGridPointsX();
        const size_t nCols = workGrid->getGridPointsY();
        const size_t totalSize = nSlices*nRows*nCols;
        const size_t totalSizeBuff = totalSize*sizeof(double);
        const size_t enqSize = totalSize/workGrid->getVectortypeAlignment();
        
        unique_ptr<CartesianOCLOOPGrid> workGridXi = grid.emptyDuplicate();
        cl_mem xiBuff = workGridXi->getRealGPUBuffer();
        
        calcXiGPU(workGrid.get(), &xiBuff, totalSize);

        workGridXi->markRealDirty();
        workGridXi->markReciDirty();
        workGridXi->markReciGPUDirty();
        workGridXi->markRealGPUClean();
        
        cl_int err;
        cl_mem binnedBuff = clCreateBuffer(_ctx, CL_MEM_READ_WRITE, totalSizeBuff, NULL, &err);
        if(err != CL_SUCCESS){
            cerr << "ERROR in allocating GPU memory of size " << totalSizeBuff << " with error " << err << endl;
            throw runtime_error("Failed to allocate GPU memory");
        }
        
        double minXi = 0.0;
        double maxXi = 0.0;
        workGridXi->minMax(minXi, maxXi);
               
        // this allocation happens after calcXi since that one internally needs to allocate a grid to hold the gradient
        unique_ptr<CartesianOCLOOPGrid> workGridB = grid.emptyDuplicate();
        unique_ptr<CartesianOCLOOPGrid> workGridC = grid.emptyDuplicate();
        
        cl_mem gridBReal = workGridB->getRealGPUBuffer();
        cl_mem gridBReci = workGridB->getReciGPUBuffer();
        cl_mem gridCReal = workGridC->getRealGPUBuffer();
        cl_mem gridCReci = workGridC->getReciGPUBuffer();        
        
        const double upper = ceil(log(maxXi/_midXi) / log(_refRatio));
        const double lower = floor(log(minXi/_midXi) / log(_refRatio));
        const size_t nBins = round(upper-lower)+1;        
        
        // get the binning array
        auto bins = MemoryFunctions::allocateScratch(nBins);
        
        for(size_t i = 0; i < nBins; ++i){
            bins->at(i) = _midXi * pow(_refRatio,(lower + i));
        }
        
        workGridA->powGrid(_alpha);
        cl_mem densABuff = workGridA->getRealGPUBuffer();
        
        unique_ptr<CartesianOCLOOPGrid> result = grid.emptyDuplicate();
        
        result->markRealDirty();
        result->markReciDirty();
        result->markRealGPUDirty();
        result->markReciGPUClean();
        
        cl_mem resReci = result->getReciGPUBuffer();
        
        const size_t nSlicesRec = result->getReciGridPointsZ();
        const size_t nRowsRec = result->getReciGridPointsX();
        const size_t nColsRec = result->getReciGridPointsY();
        const size_t totalSizeRec = nSlicesRec*nRowsRec*nColsRec;
        const size_t halfReciMem = totalSizeRec*sizeof(double);
        
        // scratch space for the kernels
        auto k0 = MemoryFunctions::allocateScratch(nRowsRec, nColsRec, nSlicesRec);
        auto k1 = MemoryFunctions::allocateScratch(nRowsRec, nColsRec, nSlicesRec);
        
        cl_mem k0Buff = clCreateBuffer(_ctx, CL_MEM_READ_WRITE, halfReciMem, NULL, &err);
        if(err != CL_SUCCESS){
            cerr << "ERROR in allocating GPU memory of size " << halfReciMem << " with error " << err << endl;
            throw runtime_error("Failed to allocate GPU memory");
        }
        cl_mem k1Buff = clCreateBuffer(_ctx, CL_MEM_READ_WRITE, halfReciMem, NULL, &err);
        if(err != CL_SUCCESS){
            cerr << "ERROR in allocating GPU memory of size " << halfReciMem << " with error " << err << endl;
            throw runtime_error("Failed to allocate GPU memory");
        }
        
        const cube* gNorms = workGridA->getGNorms();
        cl_mem gNormsBuff = clCreateBuffer(_ctx, CL_MEM_READ_ONLY, halfReciMem, NULL, &err);
        if(err != CL_SUCCESS){
            cerr << "ERROR in allocating GPU memory of size " << halfReciMem << " with error " << err << endl;
            throw runtime_error("Failed to allocate GPU memory");
        }
        err = clEnqueueWriteBuffer(_queue, gNormsBuff, CL_FALSE, 0, halfReciMem, gNorms->memptr(), 0, NULL, NULL );
        if(err != CL_SUCCESS){
            cerr << "ERROR to write gNorms data to GPU " << err << endl;
            throw runtime_error("Failed to write gNorms data to GPU.");
        }
        
        // go over all the bins
        for(size_t bin = 0; bin < nBins-1; ++bin){
            
            const double binI = bins->at(bin);
            const double binI1 = bins->at(bin+1);        
            const double incr = binI1 - binI;
            
            // get the grid points with density value in this bin
            err  = clSetKernelArg(binKernel, 0, sizeof(cl_mem), &xiBuff);
            err |= clSetKernelArg(binKernel, 1, sizeof(cl_mem), &densBuff);
            err |= clSetKernelArg(binKernel, 2, sizeof(cl_mem), &binnedBuff);
            err |= clSetKernelArg(binKernel, 3, sizeof(cl_mem), &densABuff);
            err |= clSetKernelArg(binKernel, 4, sizeof(double), &_cutoffDens);
            err |= clSetKernelArg(binKernel, 5, sizeof(double), &binI);
            err |= clSetKernelArg(binKernel, 6, sizeof(double), &binI1);
            if(err != CL_SUCCESS){
                cerr << "ERROR in setting a binning kernel argument: " << err << endl;
                throw runtime_error("Failed to set a binning kernel argument.");
            }
            
            // enqueue the kernel over the entire "1D'd" grid
            err = clEnqueueNDRangeKernel(_queue, binKernel, 1, NULL, &totalSize, NULL, 0, NULL, NULL);
            if (err != CL_SUCCESS){
                throw runtime_error("1D-enqueue of binning kernel failed.");
            }
                        
            // first round of kernels 
            if(bin == 0){
                
                if(false){
                
                    // we only need to compute these kernels in the first iteration
                    // subsequent iterations can simple use the older ones
                    calcHC10Kernels(binI, k0.get(), k1.get(), gNorms);
                
                    // transfer to GPU
                    err = clEnqueueWriteBuffer(_queue, k0Buff, CL_FALSE, 0, halfReciMem, k0->memptr(), 0, NULL, NULL );
                    if(err != CL_SUCCESS){
                        cerr << "ERROR to write kernel 0 data to GPU " << err << endl;
                        throw runtime_error("Failed to write kernel 0 data to GPU.");
                    }
                    err = clEnqueueWriteBuffer(_queue, k1Buff, CL_FALSE, 0, halfReciMem, k1->memptr(), 0, NULL, NULL );
                    if(err != CL_SUCCESS){
                        cerr << "ERROR to write kernel 1 data to GPU " << err << endl;
                        throw runtime_error("Failed to write kernel 1 data to GPU.");
                    }
                    
                } else {
                    
                    err  = clSetKernelArg(hc10SimpleKernel, 0, sizeof(cl_mem), &k0Buff);
                    err |= clSetKernelArg(hc10SimpleKernel, 1, sizeof(cl_mem), &k1Buff);
                    err |= clSetKernelArg(hc10SimpleKernel, 2, sizeof(cl_mem), &gNormsBuff);
                    err |= clSetKernelArg(hc10SimpleKernel, 3, sizeof(cl_mem), &w0Buff);
                    err |= clSetKernelArg(hc10SimpleKernel, 4, sizeof(cl_mem), &w1Buff);
                    err |= clSetKernelArg(hc10SimpleKernel, 5, sizeof(double), &binI);
                    const double xi3 = binI*binI*binI;
                    err |= clSetKernelArg(hc10SimpleKernel, 6, sizeof(double), &xi3);
                    const double t8xi3 = 8*xi3;
                    err |= clSetKernelArg(hc10SimpleKernel, 7, sizeof(double), &t8xi3);
                    err |= clSetKernelArg(hc10SimpleKernel, 8, sizeof(double), &_etaStep);
                    const int numEta = _numEta;
                    err |= clSetKernelArg(hc10SimpleKernel, 9, sizeof(int), &numEta);
                    if(err != CL_SUCCESS){
                        cerr << "ERROR in setting a HC10 simple assembly kernel argument: " << err << endl;
                        throw runtime_error("Failed to set a HC10 simple assembly kernel argument.");
                    }
                    
                    // enqueue the kernel over the entire "1D'd" grid
                    err = clEnqueueNDRangeKernel(_queue, hc10SimpleKernel, 1, NULL, &totalSizeRec, NULL, 0, NULL, NULL);
                    if (err != CL_SUCCESS){
                        throw runtime_error("1D-enqueue of simple assembly kernel failed.");
                    }                    
                }
            }
                                    
            // assemble first part of the "energy grid"
            workGridB->markRealDirty();
            workGridB->markReciDirty();
            workGridB->markReciGPUDirty();
            workGridB->markRealGPUClean();
            
            workGridC->markRealDirty();
            workGridC->markReciDirty();
            workGridC->markReciGPUDirty();
            workGridC->markRealGPUClean();
            
            err  = clSetKernelArg(hcE1Kernel, 0, sizeof(cl_mem), &gridBReal);
            err |= clSetKernelArg(hcE1Kernel, 1, sizeof(cl_mem), &gridCReal);
            err |= clSetKernelArg(hcE1Kernel, 2, sizeof(cl_mem), &xiBuff);
            err |= clSetKernelArg(hcE1Kernel, 3, sizeof(cl_mem), &binnedBuff);
            err |= clSetKernelArg(hcE1Kernel, 4, sizeof(double), &incr);
            err |= clSetKernelArg(hcE1Kernel, 5, sizeof(double), &binI);
            if(err != CL_SUCCESS){
                cerr << "ERROR in setting a HC10 energy 1 kernel argument: " << err << endl;
                throw runtime_error("Failed to set a HC10 energy 1 kernel argument.");
            }
            
            // enqueue the kernel over the entire "1D'd" grid
            err = clEnqueueNDRangeKernel(_queue, hcE1Kernel, 1, NULL, &enqSize, NULL, 0, NULL, NULL);
            if (err != CL_SUCCESS){
                throw runtime_error("1D-enqueue of HC10 energy 1 kernel failed.");
            }
                        
            // get it to g space
            workGridB->enqueueForwardTransform();
            workGridC->enqueueForwardTransform();
            
            if(bin == 0){
                err  = clSetKernelArg(hcE3ZeroKernel, 0, sizeof(cl_mem), &resReci);
                err |= clSetKernelArg(hcE3ZeroKernel, 1, sizeof(cl_mem), &k0Buff);
                err |= clSetKernelArg(hcE3ZeroKernel, 2, sizeof(cl_mem), &gridBReci);
                err |= clSetKernelArg(hcE3ZeroKernel, 3, sizeof(cl_mem), &k1Buff);
                err |= clSetKernelArg(hcE3ZeroKernel, 4, sizeof(cl_mem), &gridCReci);
                err |= clSetKernelArg(hcE3ZeroKernel, 5, sizeof(double), &incr);
                err |= clSetKernelArg(hcE3ZeroKernel, 6, sizeof(double), &binI);
                if(err != CL_SUCCESS){
                    cerr << "ERROR in setting a HC10 energy 3 zero kernel argument: " << err << endl;
                    throw runtime_error("Failed to set a HC10 energy 3 zero kernel argument.");
                }
        
                // enqueue the kernel over the entire "1D'd" grid
                err = clEnqueueNDRangeKernel(_queue, hcE3ZeroKernel, 1, NULL, &totalSizeRec, NULL, 0, NULL, NULL);
                if (err != CL_SUCCESS){
                    throw runtime_error("1D-enqueue of HC10 energy 3 zero kernel failed.");
                }
            } else {
                err  = clSetKernelArg(hcE3Kernel, 0, sizeof(cl_mem), &resReci);
                err |= clSetKernelArg(hcE3Kernel, 1, sizeof(cl_mem), &k0Buff);
                err |= clSetKernelArg(hcE3Kernel, 2, sizeof(cl_mem), &gridBReci);
                err |= clSetKernelArg(hcE3Kernel, 3, sizeof(cl_mem), &k1Buff);
                err |= clSetKernelArg(hcE3Kernel, 4, sizeof(cl_mem), &gridCReci);
                err |= clSetKernelArg(hcE3Kernel, 5, sizeof(double), &incr);
                err |= clSetKernelArg(hcE3Kernel, 6, sizeof(double), &binI);
                if(err != CL_SUCCESS){
                    cerr << "ERROR in setting a HC10 energy 3 kernel (I) e argument: " << err << endl;
                    throw runtime_error("Failed to set a HC10 energy 3 kernel (I) e argument.");
                }
        
                // enqueue the kernel over the entire "1D'd" grid
                err = clEnqueueNDRangeKernel(_queue, hcE3Kernel, 1, NULL, &totalSizeRec, NULL, 0, NULL, NULL);
                if (err != CL_SUCCESS){
                    throw runtime_error("1D-enqueue of HC10 energy 3 kernel (I) e failed.");
                }    
            }
            
            // second round of kernels
            if(false){
                calcHC10Kernels(binI1, k0.get(), k1.get(), gNorms);
            
                // transfer to GPU
                err = clEnqueueWriteBuffer(_queue, k0Buff, CL_FALSE, 0, halfReciMem, k0->memptr(), 0, NULL, NULL );
                if(err != CL_SUCCESS){
                    cerr << "ERROR to write kernel 0 data to GPU " << err << endl;
                    throw runtime_error("Failed to write kernel 0 data to GPU.");
                }
                err = clEnqueueWriteBuffer(_queue, k1Buff, CL_FALSE, 0, halfReciMem, k1->memptr(), 0, NULL, NULL );
                if(err != CL_SUCCESS){
                    cerr << "ERROR to write kernel 1 data to GPU " << err << endl;
                    throw runtime_error("Failed to write kernel 1 data to GPU.");
                }
            
            } else {
                    
                err  = clSetKernelArg(hc10SimpleKernel, 0, sizeof(cl_mem), &k0Buff);
                err |= clSetKernelArg(hc10SimpleKernel, 1, sizeof(cl_mem), &k1Buff);
                err |= clSetKernelArg(hc10SimpleKernel, 2, sizeof(cl_mem), &gNormsBuff);
                err |= clSetKernelArg(hc10SimpleKernel, 3, sizeof(cl_mem), &w0Buff);
                err |= clSetKernelArg(hc10SimpleKernel, 4, sizeof(cl_mem), &w1Buff);
                err |= clSetKernelArg(hc10SimpleKernel, 5, sizeof(double), &binI1);
                const double xi3 = binI1*binI1*binI1;
                err |= clSetKernelArg(hc10SimpleKernel, 6, sizeof(double), &xi3);
                const double t8xi3 = 8*xi3;
                err |= clSetKernelArg(hc10SimpleKernel, 7, sizeof(double), &t8xi3);
                err |= clSetKernelArg(hc10SimpleKernel, 8, sizeof(double), &_etaStep);
                const int numEta = _numEta;
                err |= clSetKernelArg(hc10SimpleKernel, 9, sizeof(int), &numEta);
                if(err != CL_SUCCESS){
                    cerr << "ERROR in setting a HC10 simple assembly kernel argument: " << err << endl;
                    throw runtime_error("Failed to set a HC10 simple assembly kernel argument.");
                }
                    
                // enqueue the kernel over the entire "1D'd" grid
                err = clEnqueueNDRangeKernel(_queue, hc10SimpleKernel, 1, NULL, &totalSizeRec, NULL, 0, NULL, NULL);
                if (err != CL_SUCCESS){
                    throw runtime_error("1D-enqueue of simple assembly kernel failed.");
                }                    
            }
                        
            // assemble second part of the "energy grid"
            workGridB->markRealDirty();
            workGridB->markReciDirty();
            workGridB->markReciGPUDirty();
            workGridB->markRealGPUClean();
            
            workGridC->markRealDirty();
            workGridC->markReciDirty();
            workGridC->markReciGPUDirty();
            workGridC->markRealGPUClean();
            
            err  = clSetKernelArg(hcE2Kernel, 0, sizeof(cl_mem), &gridBReal);
            err |= clSetKernelArg(hcE2Kernel, 1, sizeof(cl_mem), &gridCReal);
            err |= clSetKernelArg(hcE2Kernel, 2, sizeof(cl_mem), &xiBuff);
            err |= clSetKernelArg(hcE2Kernel, 3, sizeof(cl_mem), &binnedBuff);
            err |= clSetKernelArg(hcE2Kernel, 4, sizeof(double), &incr);
            err |= clSetKernelArg(hcE2Kernel, 5, sizeof(double), &binI);
            if(err != CL_SUCCESS){
                cerr << "ERROR in setting a HC10 energy 2 kernel argument: " << err << endl;
                throw runtime_error("Failed to set a HC10 energy 2 kernel argument.");
            }
            
            // enqueue the kernel over the entire "1D'd" grid
            err = clEnqueueNDRangeKernel(_queue, hcE2Kernel, 1, NULL, &enqSize, NULL, 0, NULL, NULL);
            if (err != CL_SUCCESS){
                throw runtime_error("1D-enqueue of HC10 energy 2 kernel failed.");
            }

            // get it to g space
            workGridB->enqueueForwardTransform();
            workGridC->enqueueForwardTransform();
            
            err  = clSetKernelArg(hcE3Kernel, 0, sizeof(cl_mem), &resReci);
            err |= clSetKernelArg(hcE3Kernel, 1, sizeof(cl_mem), &k0Buff);
            err |= clSetKernelArg(hcE3Kernel, 2, sizeof(cl_mem), &gridBReci);
            err |= clSetKernelArg(hcE3Kernel, 3, sizeof(cl_mem), &k1Buff);
            err |= clSetKernelArg(hcE3Kernel, 4, sizeof(cl_mem), &gridCReci);
            err |= clSetKernelArg(hcE3Kernel, 5, sizeof(double), &incr);
            err |= clSetKernelArg(hcE3Kernel, 6, sizeof(double), &binI1);
            if(err != CL_SUCCESS){
                cerr << "ERROR in setting a HC10 energy 3 kernel (II) e argument: " << err << endl;
                throw runtime_error("Failed to set a HC10 energy 3 kernel (II) e argument.");
            }
        
            // enqueue the kernel over the entire "1D'd" grid
            err = clEnqueueNDRangeKernel(_queue, hcE3Kernel, 1, NULL, &totalSizeRec, NULL, 0, NULL, NULL);
            if (err != CL_SUCCESS){
                throw runtime_error("1D-enqueue of HC10 energy 3 kernel (II) e failed.");
            }                        
        }
        workGridA.reset();
        
        result->enqueueBackwardTransform();
        
        cl_mem resReal = result->getRealGPUBuffer();
        
        err  = clSetKernelArg(hcE4Kernel, 0, sizeof(cl_mem), &resReal);
        err |= clSetKernelArg(hcE4Kernel, 1, sizeof(cl_mem), &densBuff);
        err |= clSetKernelArg(hcE4Kernel, 2, sizeof(double), &_beta);
        if(err != CL_SUCCESS){
            cerr << "ERROR in setting a HC10 energy 4 kernel argument: " << err << endl;
            throw runtime_error("Failed to set a HC10 energy 4 kernel argument.");
        }
        
        // enqueue the kernel over the entire "1D'd" grid
        err = clEnqueueNDRangeKernel(_queue, hcE4Kernel, 1, NULL, &enqSize, NULL, 0, NULL, NULL);
        if (err != CL_SUCCESS){
            throw runtime_error("1D-enqueue of HC10 energy 4 kernel failed.");
        }
        
        const double integral = result->integrate();
        
        clReleaseMemObject(k0Buff);
        clReleaseMemObject(k1Buff);
        clReleaseMemObject(binnedBuff);
        clReleaseMemObject(gNormsBuff);

        workGridXi.reset();
        workGrid.reset();
        workGridB.reset();
        workGridC.reset();
        result.reset();
        
        return _CTF*_c*integral;        
    }
    
    double calcPotential(const CartesianOCLOOPGrid& grid, CartesianOCLOOPGrid& potential) const {
        
        unique_ptr<CartesianOCLOOPGrid> workGrid = grid.duplicate();
        workGrid->transferRealToGPU();
        cl_mem densBuff = workGrid->getRealGPUBuffer();
        
        cl_int err;
        
        unique_ptr<CartesianOCLOOPGrid> workGridA = grid.duplicate();
        
        const size_t nSlices = workGrid->getGridPointsZ();
        const size_t nRows = workGrid->getGridPointsX();
        const size_t nCols = workGrid->getGridPointsY();
        const size_t totalSize = nSlices*nRows*nCols;
        const size_t totalSizeBuff = totalSize*sizeof(double);
        const size_t enqSize = totalSize/workGrid->getVectortypeAlignment();
        
        unique_ptr<CartesianOCLOOPGrid> workGridXi = grid.emptyDuplicate();
        cl_mem xiBuff = workGridXi->getRealGPUBuffer();
        
        cl_mem ssBuff = clCreateBuffer(_ctx, CL_MEM_READ_WRITE, totalSizeBuff, NULL, &err);
        if(err != CL_SUCCESS){
            cerr << "ERROR in allocating GPU memory of size " << totalSizeBuff << " with error " << err << endl;
            throw runtime_error("Failed to allocate GPU memory");
        }
        
        calcXiAndSSGPU(workGrid.get(), &xiBuff, &ssBuff, totalSize);

        workGridXi->markRealDirty();
        workGridXi->markReciDirty();
        workGridXi->markReciGPUDirty();
        workGridXi->markRealGPUClean();

        double minXi = 0.0;
        double maxXi = 0.0;
        workGridXi->minMax(minXi, maxXi);
                        
        cl_mem binnedBuff = clCreateBuffer(_ctx, CL_MEM_READ_WRITE, totalSizeBuff, NULL, &err);
        if(err != CL_SUCCESS){
            cerr << "ERROR in allocating GPU memory of size " << totalSizeBuff << " with error " << err << endl;
            throw runtime_error("Failed to allocate GPU memory");
        }
        
        // this allocation happens after calcXi since that one internally needs to allocate a grid to hold the gradient
        unique_ptr<CartesianOCLOOPGrid> workGridB = grid.emptyDuplicate();
        unique_ptr<CartesianOCLOOPGrid> workGridC = grid.emptyDuplicate();
        
        unique_ptr<CartesianOCLOOPGrid> workGridBeta = grid.emptyDuplicate();
        
        cl_mem gridBBuff = workGridB->getRealGPUBuffer();
        cl_mem gridCBuff = workGridC->getRealGPUBuffer();
        cl_mem gridBBuffReci = workGridB->getReciGPUBuffer();
        cl_mem gridCBuffReci = workGridC->getReciGPUBuffer();
        cl_mem gridBetaBuff = workGridBeta->getRealGPUBuffer();
        cl_mem gridBetaBuffReci = workGridBeta->getReciGPUBuffer();
        cl_mem realPotBuff = potential.getRealGPUBuffer();

        err  = clSetKernelArg(hcP0Kernel, 0, sizeof(cl_mem), &realPotBuff);
        err |= clSetKernelArg(hcP0Kernel, 1, sizeof(cl_mem), &gridBetaBuff);
        err |= clSetKernelArg(hcP0Kernel, 2, sizeof(cl_mem), &densBuff);
        err |= clSetKernelArg(hcP0Kernel, 3, sizeof(double), &_beta);
        if(err != CL_SUCCESS){
            cerr << "ERROR in setting a HC potential kernel 0 argument: " << err << endl;
            throw runtime_error("Failed to set a HC potential kernel 0 argument.");
        }

        // enqueue the kernel over the entire "1D'd" grid
        err = clEnqueueNDRangeKernel(_queue, hcP0Kernel, 1, NULL, &enqSize, NULL, 0, NULL, NULL);
        if (err != CL_SUCCESS){
            throw runtime_error("1D-enqueue of HC potential kernel 0 failed.");
        }
        
        // XXX maybe removable once everything done?
        potential.markRealDirty();
        potential.markReciDirty();
        potential.markRealGPUClean();
        potential.markReciGPUDirty();
        
        workGridBeta->markRealDirty();
        workGridBeta->markReciDirty();
        workGridBeta->markRealGPUClean();
        workGridBeta->markReciGPUDirty();
        
        workGridBeta->enqueueForwardTransform();
        
        unique_ptr<CartesianOCLOOPGrid> fkf0 = grid.emptyDuplicate();
        unique_ptr<CartesianOCLOOPGrid> fk1rf = grid.emptyDuplicate();
        
        cl_mem fkf0Buff = fkf0->getRealGPUBuffer();
        cl_mem fk1rfBuff = fk1rf->getRealGPUBuffer();
        cl_mem fkf0BuffReci = fkf0->getReciGPUBuffer();
        cl_mem fk1rfBuffReci = fk1rf->getReciGPUBuffer();
        
        const double upper = ceil(log(maxXi/_midXi) / log(_refRatio));
        const double lower = floor(log(minXi/_midXi) / log(_refRatio));
        const size_t nBins = round(upper-lower)+1;        
        
        // get the binning array
        auto bins = MemoryFunctions::allocateScratch(nBins);
        
        for(size_t i = 0; i < nBins; ++i){
            bins->at(i) = _midXi * pow(_refRatio,(lower + i));
        }
        
        workGridA->powGrid(_alpha);
        cl_mem densABuff = workGridA->getRealGPUBuffer();
        
        unique_ptr<CartesianOCLOOPGrid> result = grid.emptyDuplicate();
        
        result->markRealDirty();
        result->markReciDirty();
        result->markRealGPUDirty();
        result->markReciGPUClean();
        
        cl_mem resReci = result->getReciGPUBuffer();
        
        const size_t nSlicesRec = result->getReciGridPointsZ();
        const size_t nRowsRec = result->getReciGridPointsX();
        const size_t nColsRec = result->getReciGridPointsY();
        const size_t totalSizeRec = nSlicesRec*nRowsRec*nColsRec;
        const size_t halfReciMem = totalSizeRec*sizeof(double);
        const size_t totalReciMem = halfReciMem*2;
        
        // scratch space for the kernels
        auto k0 = MemoryFunctions::allocateScratch(nRowsRec, nColsRec, nSlicesRec);
        auto k1 = MemoryFunctions::allocateScratch(nRowsRec, nColsRec, nSlicesRec);
        
        cl_mem k0Buff = clCreateBuffer(_ctx, CL_MEM_READ_WRITE, halfReciMem, NULL, &err);
        if(err != CL_SUCCESS){
            cerr << "ERROR in allocating GPU memory of size " << halfReciMem << " with error " << err << endl;
            throw runtime_error("Failed to allocate GPU memory");
        }
        cl_mem k1Buff = clCreateBuffer(_ctx, CL_MEM_READ_WRITE, halfReciMem, NULL, &err);
        if(err != CL_SUCCESS){
            cerr << "ERROR in allocating GPU memory of size " << halfReciMem << " with error " << err << endl;
            throw runtime_error("Failed to allocate GPU memory");
        }
        
        // other scratch space        
        cl_mem intpotBuff = clCreateBuffer(_ctx, CL_MEM_READ_WRITE, totalSizeBuff, NULL, &err);
        if(err != CL_SUCCESS){
            cerr << "ERROR in allocating GPU memory of size " << halfReciMem << " with error " << err << endl;
            throw runtime_error("Failed to allocate GPU memory");
        }
        cl_mem pottmpBuff = clCreateBuffer(_ctx, CL_MEM_READ_WRITE, totalSizeBuff, NULL, &err);
        if(err != CL_SUCCESS){
            cerr << "ERROR in allocating GPU memory of size " << halfReciMem << " with error " << err << endl;
            throw runtime_error("Failed to allocate GPU memory");
        }
        
        const cube* gNorms = workGridA->getGNorms();
        cl_mem gNormsBuff = clCreateBuffer(_ctx, CL_MEM_READ_ONLY, halfReciMem, NULL, &err);
        if(err != CL_SUCCESS){
            cerr << "ERROR in allocating GPU memory of size " << halfReciMem << " with error " << err << endl;
            throw runtime_error("Failed to allocate GPU memory");
        }
        err = clEnqueueWriteBuffer(_queue, gNormsBuff, CL_FALSE, 0, halfReciMem, gNorms->memptr(), 0, NULL, NULL );
        if(err != CL_SUCCESS){
            cerr << "ERROR to write gNorms data to GPU " << err << endl;
            throw runtime_error("Failed to write gNorms data to GPU.");
        }
        
        // go over all the bins
        for(size_t bin = 0; bin < nBins-1; ++bin){
            
            const double binI = bins->at(bin);
            const double binI1 = bins->at(bin+1);        
            const double incr = binI1 - binI;
            
            // get the grid points with density value in this bin
            err  = clSetKernelArg(binKernel, 0, sizeof(cl_mem), &xiBuff);
            err |= clSetKernelArg(binKernel, 1, sizeof(cl_mem), &densBuff);
            err |= clSetKernelArg(binKernel, 2, sizeof(cl_mem), &binnedBuff);
            err |= clSetKernelArg(binKernel, 3, sizeof(cl_mem), &densABuff);
            err |= clSetKernelArg(binKernel, 4, sizeof(double), &_cutoffDens);
            err |= clSetKernelArg(binKernel, 5, sizeof(double), &binI);
            err |= clSetKernelArg(binKernel, 6, sizeof(double), &binI1);
            if(err != CL_SUCCESS){
                cerr << "ERROR in setting a binning kernel argument: " << err << endl;
                throw runtime_error("Failed to set a binning kernel argument.");
            }
            
            // enqueue the kernel over the entire "1D'd" grid
            err = clEnqueueNDRangeKernel(_queue, binKernel, 1, NULL, &totalSize, NULL, 0, NULL, NULL);
            if (err != CL_SUCCESS){
                throw runtime_error("1D-enqueue of binning kernel failed.");
            }
            
            // first round of kernels 
            if(bin == 0){
                // we only need to compute these kernels in the first iteration
                // subsequent iterations can simple use the older ones
                if(false){
                    calcHC10Kernels(binI, k0.get(), k1.get(), gNorms);
            
                    // transfer to GPU
                    err = clEnqueueWriteBuffer(_queue, k0Buff, CL_FALSE, 0, halfReciMem, k0->memptr(), 0, NULL, NULL );
                    if(err != CL_SUCCESS){
                        cerr << "ERROR to write kernel 0 data to GPU " << err << endl;
                        throw runtime_error("Failed to write kernel 0 data to GPU.");
                    }
                    err = clEnqueueWriteBuffer(_queue, k1Buff, CL_FALSE, 0, halfReciMem, k1->memptr(), 0, NULL, NULL );
                    if(err != CL_SUCCESS){
                        cerr << "ERROR to write kernel 1 data to GPU " << err << endl;
                        throw runtime_error("Failed to write kernel 1 data to GPU.");
                    }
                } else {
                    
                    err  = clSetKernelArg(hc10SimpleKernel, 0, sizeof(cl_mem), &k0Buff);
                    err |= clSetKernelArg(hc10SimpleKernel, 1, sizeof(cl_mem), &k1Buff);
                    err |= clSetKernelArg(hc10SimpleKernel, 2, sizeof(cl_mem), &gNormsBuff);
                    err |= clSetKernelArg(hc10SimpleKernel, 3, sizeof(cl_mem), &w0Buff);
                    err |= clSetKernelArg(hc10SimpleKernel, 4, sizeof(cl_mem), &w1Buff);
                    err |= clSetKernelArg(hc10SimpleKernel, 5, sizeof(double), &binI);
                    const double xi3 = binI*binI*binI;
                    err |= clSetKernelArg(hc10SimpleKernel, 6, sizeof(double), &xi3);
                    const double t8xi3 = 8*xi3;
                    err |= clSetKernelArg(hc10SimpleKernel, 7, sizeof(double), &t8xi3);
                    err |= clSetKernelArg(hc10SimpleKernel, 8, sizeof(double), &_etaStep);
                    const int numEta = _numEta;
                    err |= clSetKernelArg(hc10SimpleKernel, 9, sizeof(int), &numEta);
                    if(err != CL_SUCCESS){
                        cerr << "ERROR in setting a HC10 simple assembly kernel argument: " << err << endl;
                        throw runtime_error("Failed to set a HC10 simple assembly kernel argument.");
                    }
                    
                    // enqueue the kernel over the entire "1D'd" grid
                    err = clEnqueueNDRangeKernel(_queue, hc10SimpleKernel, 1, NULL, &totalSizeRec, NULL, 0, NULL, NULL);
                    if (err != CL_SUCCESS){
                        throw runtime_error("1D-enqueue of simple assembly kernel failed.");
                    }
                }
                
                fkf0->markRealDirty();
                fkf0->markReciDirty();
                fkf0->markRealGPUDirty();
                fkf0->markReciGPUClean();
                
                fk1rf->markRealDirty();
                fk1rf->markReciDirty();
                fk1rf->markRealGPUDirty();
                fk1rf->markReciGPUClean();
                
                err  = clSetKernelArg(hcP1Kernel, 0, sizeof(cl_mem), &fkf0BuffReci);
                err |= clSetKernelArg(hcP1Kernel, 1, sizeof(cl_mem), &fk1rfBuffReci);
                err |= clSetKernelArg(hcP1Kernel, 2, sizeof(cl_mem), &k0Buff);
                err |= clSetKernelArg(hcP1Kernel, 3, sizeof(cl_mem), &k1Buff);
                err |= clSetKernelArg(hcP1Kernel, 4, sizeof(cl_mem), &gridBetaBuffReci);
                err |= clSetKernelArg(hcP1Kernel, 5, sizeof(double), &binI);
                if(err != CL_SUCCESS){
                    cerr << "ERROR in setting a HC potential kernel 1 argument: " << err << endl;
                    throw runtime_error("Failed to set a HC potential kernel 1 argument.");
                }
                
                // enqueue the kernel over the entire "1D'd" grid
                err = clEnqueueNDRangeKernel(_queue, hcP1Kernel, 1, NULL, &totalSizeRec, NULL, 0, NULL, NULL);
                if (err != CL_SUCCESS){
                    throw runtime_error("1D-enqueue of HC potential kernel 1 failed.");
                }                
                
                fkf0->enqueueBackwardTransform();
                fk1rf->enqueueBackwardTransform();
            }
          
            // assemble first part of the "energy grid"
            workGridB->markRealDirty();
            workGridB->markReciDirty();
            workGridB->markReciGPUDirty();
            workGridB->markRealGPUClean();

            workGridC->markRealDirty();
            workGridC->markReciDirty();
            workGridC->markReciGPUDirty();
            workGridC->markRealGPUClean();

            if(bin == 0){
                err  = clSetKernelArg(hcP2ZeroKernel, 0, sizeof(cl_mem), &xiBuff);
                err |= clSetKernelArg(hcP2ZeroKernel, 1, sizeof(cl_mem), &gridBBuff);
                err |= clSetKernelArg(hcP2ZeroKernel, 2, sizeof(cl_mem), &gridCBuff);
                err |= clSetKernelArg(hcP2ZeroKernel, 3, sizeof(cl_mem), &binnedBuff);
                err |= clSetKernelArg(hcP2ZeroKernel, 4, sizeof(cl_mem), &densBuff);
                err |= clSetKernelArg(hcP2ZeroKernel, 5, sizeof(cl_mem), &intpotBuff);
                err |= clSetKernelArg(hcP2ZeroKernel, 6, sizeof(cl_mem), &pottmpBuff);
                err |= clSetKernelArg(hcP2ZeroKernel, 7, sizeof(cl_mem), &fkf0Buff);
                err |= clSetKernelArg(hcP2ZeroKernel, 8, sizeof(cl_mem), &fk1rfBuff);
                err |= clSetKernelArg(hcP2ZeroKernel, 9, sizeof(double), &binI);
                err |= clSetKernelArg(hcP2ZeroKernel, 10, sizeof(double), &binI1);
                err |= clSetKernelArg(hcP2ZeroKernel, 11, sizeof(double), &_cutoffDens);
                err |= clSetKernelArg(hcP2ZeroKernel, 12, sizeof(double), &incr);
                if(err != CL_SUCCESS){
                    cerr << "ERROR in setting a HC potential kernel 2 (zero) argument: " << err << endl;
                    throw runtime_error("Failed to set a HC potential kernel 2 (zero) argument.");
                }

                // enqueue the kernel over the entire "1D'd" grid
                err = clEnqueueNDRangeKernel(_queue, hcP2ZeroKernel, 1, NULL, &totalSize, NULL, 0, NULL, NULL);
                if (err != CL_SUCCESS){
                    throw runtime_error("1D-enqueue of HC potential kernel 2 (zero) failed.");
                }
            } else {
                err  = clSetKernelArg(hcP2Kernel, 0, sizeof(cl_mem), &xiBuff);
                err |= clSetKernelArg(hcP2Kernel, 1, sizeof(cl_mem), &gridBBuff);
                err |= clSetKernelArg(hcP2Kernel, 2, sizeof(cl_mem), &gridCBuff);
                err |= clSetKernelArg(hcP2Kernel, 3, sizeof(cl_mem), &binnedBuff);
                err |= clSetKernelArg(hcP2Kernel, 4, sizeof(cl_mem), &densBuff);
                err |= clSetKernelArg(hcP2Kernel, 5, sizeof(cl_mem), &intpotBuff);
                err |= clSetKernelArg(hcP2Kernel, 6, sizeof(cl_mem), &pottmpBuff);
                err |= clSetKernelArg(hcP2Kernel, 7, sizeof(cl_mem), &fkf0Buff);
                err |= clSetKernelArg(hcP2Kernel, 8, sizeof(cl_mem), &fk1rfBuff);
                err |= clSetKernelArg(hcP2Kernel, 9, sizeof(double), &binI);
                err |= clSetKernelArg(hcP2Kernel, 10, sizeof(double), &binI1);
                err |= clSetKernelArg(hcP2Kernel, 11, sizeof(double), &_cutoffDens);
                err |= clSetKernelArg(hcP2Kernel, 12, sizeof(double), &incr);
                if(err != CL_SUCCESS){
                    cerr << "ERROR in setting a HC potential kernel 2 argument: " << err << endl;
                    throw runtime_error("Failed to set a HC potential kernel 2 argument.");
                }

                // enqueue the kernel over the entire "1D'd" grid
                err = clEnqueueNDRangeKernel(_queue, hcP2Kernel, 1, NULL, &totalSize, NULL, 0, NULL, NULL);
                if (err != CL_SUCCESS){
                    throw runtime_error("1D-enqueue of HC potential kernel 2 failed.");
                }
            }
                        
            // get it to g space
            workGridB->enqueueForwardTransform();
            workGridC->enqueueForwardTransform();
            
            if(bin == 0){
                err  = clSetKernelArg(hcE3ZeroKernel, 0, sizeof(cl_mem), &resReci);
                err |= clSetKernelArg(hcE3ZeroKernel, 1, sizeof(cl_mem), &k0Buff);
                err |= clSetKernelArg(hcE3ZeroKernel, 2, sizeof(cl_mem), &gridBBuffReci);
                err |= clSetKernelArg(hcE3ZeroKernel, 3, sizeof(cl_mem), &k1Buff);
                err |= clSetKernelArg(hcE3ZeroKernel, 4, sizeof(cl_mem), &gridCBuffReci);
                err |= clSetKernelArg(hcE3ZeroKernel, 5, sizeof(double), &incr);
                err |= clSetKernelArg(hcE3ZeroKernel, 6, sizeof(double), &binI);
                if(err != CL_SUCCESS){
                    cerr << "ERROR in setting a HC10 energy 3 zero kernel argument: " << err << endl;
                    throw runtime_error("Failed to set a HC10 energy 3 zero kernel argument.");
                }
        
                // enqueue the kernel over the entire "1D'd" grid
                err = clEnqueueNDRangeKernel(_queue, hcE3ZeroKernel, 1, NULL, &totalSizeRec, NULL, 0, NULL, NULL);
                if (err != CL_SUCCESS){
                    throw runtime_error("1D-enqueue of HC10 energy 3 zero kernel failed.");
                }
            } else {
                err  = clSetKernelArg(hcE3Kernel, 0, sizeof(cl_mem), &resReci);
                err |= clSetKernelArg(hcE3Kernel, 1, sizeof(cl_mem), &k0Buff);
                err |= clSetKernelArg(hcE3Kernel, 2, sizeof(cl_mem), &gridBBuffReci);
                err |= clSetKernelArg(hcE3Kernel, 3, sizeof(cl_mem), &k1Buff);
                err |= clSetKernelArg(hcE3Kernel, 4, sizeof(cl_mem), &gridCBuffReci);
                err |= clSetKernelArg(hcE3Kernel, 5, sizeof(double), &incr);
                err |= clSetKernelArg(hcE3Kernel, 6, sizeof(double), &binI);
                if(err != CL_SUCCESS){
                    cerr << "ERROR in setting a HC10 energy 3 kernel (I) p argument: " << err << endl;
                    throw runtime_error("Failed to set a HC10 energy 3 kernel (I) p argument.");
                }
        
                // enqueue the kernel over the entire "1D'd" grid
                err = clEnqueueNDRangeKernel(_queue, hcE3Kernel, 1, NULL, &totalSizeRec, NULL, 0, NULL, NULL);
                if (err != CL_SUCCESS){
                    throw runtime_error("1D-enqueue of HC10 energy 3 kernel (I) p failed.");
                }    
            }
        
            // second round of kernels
            if(false){
                calcHC10Kernels(binI1, k0.get(), k1.get(), gNorms);
                
                // transfer to GPU
                err = clEnqueueWriteBuffer(_queue, k0Buff, CL_FALSE, 0, halfReciMem, k0->memptr(), 0, NULL, NULL );
                if(err != CL_SUCCESS){
                    cerr << "ERROR to write kernel 0 data to GPU " << err << endl;
                    throw runtime_error("Failed to write kernel 0 data to GPU.");
                }
                err = clEnqueueWriteBuffer(_queue, k1Buff, CL_FALSE, 0, halfReciMem, k1->memptr(), 0, NULL, NULL );
                if(err != CL_SUCCESS){
                    cerr << "ERROR to write kernel 1 data to GPU " << err << endl;
                    throw runtime_error("Failed to write kernel 1 data to GPU.");
                }
            } else {
                
                err  = clSetKernelArg(hc10SimpleKernel, 0, sizeof(cl_mem), &k0Buff);
                err |= clSetKernelArg(hc10SimpleKernel, 1, sizeof(cl_mem), &k1Buff);
                err |= clSetKernelArg(hc10SimpleKernel, 2, sizeof(cl_mem), &gNormsBuff);
                err |= clSetKernelArg(hc10SimpleKernel, 3, sizeof(cl_mem), &w0Buff);
                err |= clSetKernelArg(hc10SimpleKernel, 4, sizeof(cl_mem), &w1Buff);
                err |= clSetKernelArg(hc10SimpleKernel, 5, sizeof(double), &binI1);
                const double xi3 = binI1*binI1*binI1;
                err |= clSetKernelArg(hc10SimpleKernel, 6, sizeof(double), &xi3);
                const double t8xi3 = 8*xi3;
                err |= clSetKernelArg(hc10SimpleKernel, 7, sizeof(double), &t8xi3);
                err |= clSetKernelArg(hc10SimpleKernel, 8, sizeof(double), &_etaStep);
                const int numEta = _numEta;
                err |= clSetKernelArg(hc10SimpleKernel, 9, sizeof(int), &numEta);
                if(err != CL_SUCCESS){
                    cerr << "ERROR in setting a HC10 simple assembly kernel argument: " << err << endl;
                    throw runtime_error("Failed to set a HC10 simple assembly kernel argument.");
                }
                    
                // enqueue the kernel over the entire "1D'd" grid
                err = clEnqueueNDRangeKernel(_queue, hc10SimpleKernel, 1, NULL, &totalSizeRec, NULL, 0, NULL, NULL);
                if (err != CL_SUCCESS){
                    throw runtime_error("1D-enqueue of simple assembly kernel failed.");
                }
            }
            
            fkf0->markRealDirty();
            fkf0->markReciDirty();
            fkf0->markRealGPUDirty();
            fkf0->markReciGPUClean();
            
            fk1rf->markRealDirty();
            fk1rf->markReciDirty();
            fk1rf->markRealGPUDirty();
            fk1rf->markReciGPUClean();
                
            err  = clSetKernelArg(hcP1Kernel, 0, sizeof(cl_mem), &fkf0BuffReci);
            err |= clSetKernelArg(hcP1Kernel, 1, sizeof(cl_mem), &fk1rfBuffReci);
            err |= clSetKernelArg(hcP1Kernel, 2, sizeof(cl_mem), &k0Buff);
            err |= clSetKernelArg(hcP1Kernel, 3, sizeof(cl_mem), &k1Buff);
            err |= clSetKernelArg(hcP1Kernel, 4, sizeof(cl_mem), &gridBetaBuffReci);
            err |= clSetKernelArg(hcP1Kernel, 5, sizeof(double), &binI1);
            if(err != CL_SUCCESS){
                cerr << "ERROR in setting a HC potential kernel 1 argument: " << err << endl;
                throw runtime_error("Failed to set a HC potential kernel 1 argument.");
            }
                
            // enqueue the kernel over the entire "1D'd" grid
            err = clEnqueueNDRangeKernel(_queue, hcP1Kernel, 1, NULL, &totalSizeRec, NULL, 0, NULL, NULL);
            if (err != CL_SUCCESS){
                throw runtime_error("1D-enqueue of HC potential kernel 1 failed.");
            }
                       
            fkf0->enqueueBackwardTransform();
            fk1rf->enqueueBackwardTransform();
            
            // assemble second part of the "energy grid"
            workGridB->markRealDirty();
            workGridB->markReciDirty();
            workGridB->markReciGPUDirty();
            workGridB->markRealGPUClean();
            
            workGridC->markRealDirty();
            workGridC->markReciDirty();
            workGridC->markReciGPUDirty();
            workGridC->markRealGPUClean();
            
            err  = clSetKernelArg(hcP3Kernel, 0, sizeof(cl_mem), &xiBuff);
            err |= clSetKernelArg(hcP3Kernel, 1, sizeof(cl_mem), &gridBBuff);
            err |= clSetKernelArg(hcP3Kernel, 2, sizeof(cl_mem), &gridCBuff);
            err |= clSetKernelArg(hcP3Kernel, 3, sizeof(cl_mem), &binnedBuff);
            err |= clSetKernelArg(hcP3Kernel, 4, sizeof(cl_mem), &densBuff);
            err |= clSetKernelArg(hcP3Kernel, 5, sizeof(cl_mem), &intpotBuff);
            err |= clSetKernelArg(hcP3Kernel, 6, sizeof(cl_mem), &pottmpBuff);
            err |= clSetKernelArg(hcP3Kernel, 7, sizeof(cl_mem), &fkf0Buff);
            err |= clSetKernelArg(hcP3Kernel, 8, sizeof(cl_mem), &fk1rfBuff);
            err |= clSetKernelArg(hcP3Kernel, 9, sizeof(double), &binI);
            err |= clSetKernelArg(hcP3Kernel, 10, sizeof(double), &binI1);
            err |= clSetKernelArg(hcP3Kernel, 11, sizeof(double), &_cutoffDens);
            err |= clSetKernelArg(hcP3Kernel, 12, sizeof(double), &incr);
            if(err != CL_SUCCESS){
                cerr << "ERROR in setting a HC potential kernel 3 argument: " << err << endl;
                throw runtime_error("Failed to set a HC potential kernel 3 argument.");
            }

            // enqueue the kernel over the entire "1D'd" grid
            err = clEnqueueNDRangeKernel(_queue, hcP3Kernel, 1, NULL, &totalSize, NULL, 0, NULL, NULL);
            if (err != CL_SUCCESS){
                throw runtime_error("1D-enqueue of HC potential kernel 3 failed.");
            }
            
            workGridB->enqueueForwardTransform();
            workGridC->enqueueForwardTransform();
            
            err  = clSetKernelArg(hcE3Kernel, 0, sizeof(cl_mem), &resReci);
            err |= clSetKernelArg(hcE3Kernel, 1, sizeof(cl_mem), &k0Buff);
            err |= clSetKernelArg(hcE3Kernel, 2, sizeof(cl_mem), &gridBBuffReci);
            err |= clSetKernelArg(hcE3Kernel, 3, sizeof(cl_mem), &k1Buff);
            err |= clSetKernelArg(hcE3Kernel, 4, sizeof(cl_mem), &gridCBuffReci);
            err |= clSetKernelArg(hcE3Kernel, 5, sizeof(double), &incr);
            err |= clSetKernelArg(hcE3Kernel, 6, sizeof(double), &binI1);
            if(err != CL_SUCCESS){
                cerr << "ERROR in setting a HC10 energy 3 kernel (II) argument: " << err << endl;
                throw runtime_error("Failed to set a HC10 energy 3 kernel (II) argument.");
            }
        
            // enqueue the kernel over the entire "1D'd" grid
            err = clEnqueueNDRangeKernel(_queue, hcE3Kernel, 1, NULL, &totalSizeRec, NULL, 0, NULL, NULL);
            if (err != CL_SUCCESS){
                throw runtime_error("1D-enqueue of HC10 energy 3 kernel (II) failed.");
            }
        }
        result->markReciDirty();
        result->markRealDirty();
        result->markRealGPUDirty();
        result->markReciGPUClean();
        
        const double preFactor = _CTF*_c;
        
        result->enqueueBackwardTransform();
        
        cl_mem resBuff = result->getRealGPUBuffer();
                
        err  = clSetKernelArg(hcP4Kernel, 0, sizeof(cl_mem), &resBuff);
        err |= clSetKernelArg(hcP4Kernel, 1, sizeof(cl_mem), &densBuff);
        err |= clSetKernelArg(hcP4Kernel, 2, sizeof(cl_mem), &densABuff);
        err |= clSetKernelArg(hcP4Kernel, 3, sizeof(cl_mem), &realPotBuff);
        err |= clSetKernelArg(hcP4Kernel, 4, sizeof(cl_mem), &ssBuff);
        err |= clSetKernelArg(hcP4Kernel, 5, sizeof(cl_mem), &pottmpBuff);
        err |= clSetKernelArg(hcP4Kernel, 6, sizeof(cl_mem), &intpotBuff);
        err |= clSetKernelArg(hcP4Kernel, 7, sizeof(double), &_hc_lambda);
        err |= clSetKernelArg(hcP4Kernel, 8, sizeof(double), &_alpha);
        err |= clSetKernelArg(hcP4Kernel, 9, sizeof(double), &_beta);
        if(err != CL_SUCCESS){
            cerr << "ERROR in setting a HC potential kernel 4 argument: " << err << endl;
            throw runtime_error("Failed to set a HC potential kernel 4 argument.");
        }

        // enqueue the kernel over the entire "1D'd" grid
        err = clEnqueueNDRangeKernel(_queue, hcP4Kernel, 1, NULL, &enqSize, NULL, 0, NULL, NULL);
        if (err != CL_SUCCESS){
            throw runtime_error("1D-enqueue of HC potential kernel 4 failed.");
        }
                
        workGridA.reset();
                
        potential.markReciDirty(); //XXX
        potential.markRealDirty(); // XXX
        potential.markReciGPUDirty(); ///XXX
        potential.markRealGPUClean(); //XXX
        
        // get the grid into reciprocal space
        workGrid->markRealDirty();
        workGrid->markReciDirty();
        workGrid->markReciGPUDirty();
        workGrid->markRealGPUClean();

        workGrid->enqueueForwardTransform();
        
        cl_mem reciRefGrid = workGrid->getReciGPUBuffer();
        cl_mem reciGridB = workGridB->getReciGPUBuffer();
        cl_mem realGridB = workGridB->getRealGPUBuffer();
        
        // copy over and fix state
        err = clEnqueueCopyBuffer(_queue, reciRefGrid, reciGridB, 0, 0, totalReciMem, 0, NULL, NULL);
        if (err != CL_SUCCESS){
            throw runtime_error("Enqueue of reciprocal buffer copy failed.");
        }
        workGridB->markReciDirty();
        workGridB->markRealDirty();
        workGridB->markRealGPUDirty();
        workGridB->markReciGPUClean();
        
        workGridB->multiplyGVectorsX();
        
        // do the multiplication with ss
        workGridB->enqueueBackwardTransform();
        
        err  = clSetKernelArg(hcP5Kernel, 0, sizeof(cl_mem), &realGridB);
        err |= clSetKernelArg(hcP5Kernel, 1, sizeof(cl_mem), &ssBuff);
        if(err != CL_SUCCESS){
            cerr << "ERROR in setting a HC10 potential 5 kernel argument (I): " << err << endl;
            throw runtime_error("Failed to set a HC10 potential 5 kernel argument (I).");
        }
        
        // enqueue the kernel over the entire "1D'd" grid
        err = clEnqueueNDRangeKernel(_queue, hcP5Kernel, 1, NULL, &enqSize, NULL, 0, NULL, NULL);
        if (err != CL_SUCCESS){
            throw runtime_error("1D-enqueue of HC10 potential 5 kernel failed.");
        }
                
        workGridB->markReciGPUDirty();
        
        workGridB->multiplyGVectorsX();        
        workGridB->enqueueBackwardTransform();
        
        err  = clSetKernelArg(hcP6Kernel, 0, sizeof(cl_mem), &realPotBuff);
        err |= clSetKernelArg(hcP6Kernel, 1, sizeof(cl_mem), &realGridB);
        if(err != CL_SUCCESS){
            cerr << "ERROR in setting a HC10 potential 6 kernel argument (I): " << err << endl;
            throw runtime_error("Failed to set a HC10 potential 6 kernel argument (I).");
        }
        
        // enqueue the kernel over the entire "1D'd" grid
        err = clEnqueueNDRangeKernel(_queue, hcP6Kernel, 1, NULL, &enqSize, NULL, 0, NULL, NULL);
        if (err != CL_SUCCESS){
            throw runtime_error("1D-enqueue of HC10 potential 6 kernel failed.");
        }
        
        // NOW Y
        
        // copy over and fix state
        err = clEnqueueCopyBuffer(_queue, reciRefGrid, reciGridB, 0, 0, totalReciMem, 0, NULL, NULL);
        if (err != CL_SUCCESS){
            throw runtime_error("Enqueue of reciprocal buffer copy failed.");
        }
        workGridB->markReciDirty();
        workGridB->markRealDirty();
        workGridB->markRealGPUDirty();
        workGridB->markReciGPUClean();
        
        workGridB->multiplyGVectorsY();
        
        // do the multiplication with ss
        workGridB->enqueueBackwardTransform();
        
        err  = clSetKernelArg(hcP5Kernel, 0, sizeof(cl_mem), &realGridB);
        err |= clSetKernelArg(hcP5Kernel, 1, sizeof(cl_mem), &ssBuff);
        if(err != CL_SUCCESS){
            cerr << "ERROR in setting a HC10 potential 5 kernel argument (II): " << err << endl;
            throw runtime_error("Failed to set a HC10 potential 5 kernel argument (II).");
        }
        
        // enqueue the kernel over the entire "1D'd" grid
        err = clEnqueueNDRangeKernel(_queue, hcP5Kernel, 1, NULL, &enqSize, NULL, 0, NULL, NULL);
        if (err != CL_SUCCESS){
            throw runtime_error("1D-enqueue of HC10 potential 5 kernel failed.");
        }
        
        workGridB->markReciGPUDirty();
        
        workGridB->multiplyGVectorsY();        
        workGridB->enqueueBackwardTransform();
        
        err  = clSetKernelArg(hcP6Kernel, 0, sizeof(cl_mem), &realPotBuff);
        err |= clSetKernelArg(hcP6Kernel, 1, sizeof(cl_mem), &realGridB);
        if(err != CL_SUCCESS){
            cerr << "ERROR in setting a HC10 potential 6 kernel argument: " << err << endl;
            throw runtime_error("Failed to set a HC10 potential 6 kernel argument.");
        }
        
        // enqueue the kernel over the entire "1D'd" grid
        err = clEnqueueNDRangeKernel(_queue, hcP6Kernel, 1, NULL, &enqSize, NULL, 0, NULL, NULL);
        if (err != CL_SUCCESS){
            throw runtime_error("1D-enqueue of HC10 potential 6 kernel failed.");
        }
        
        // NOW Z
        
        // copy over and fix state
        err = clEnqueueCopyBuffer(_queue, reciRefGrid, reciGridB, 0, 0, totalReciMem, 0, NULL, NULL);
        if (err != CL_SUCCESS){
            throw runtime_error("Enqueue of reciprocal buffer copy failed.");
        }
        workGridB->markReciDirty();
        workGridB->markRealDirty();
        workGridB->markRealGPUDirty();
        workGridB->markReciGPUClean();
        
        workGridB->multiplyGVectorsZ();
        
        // do the multiplication with ss
        workGridB->enqueueBackwardTransform();
        
        err  = clSetKernelArg(hcP5Kernel, 0, sizeof(cl_mem), &realGridB);
        err |= clSetKernelArg(hcP5Kernel, 1, sizeof(cl_mem), &ssBuff);
        if(err != CL_SUCCESS){
            cerr << "ERROR in setting a HC10 potential 5 kernel argument (III): " << err << endl;
            throw runtime_error("Failed to set a HC10 potential 5 kernel argument (III).");
        }
        
        // enqueue the kernel over the entire "1D'd" grid
        err = clEnqueueNDRangeKernel(_queue, hcP5Kernel, 1, NULL, &enqSize, NULL, 0, NULL, NULL);
        if (err != CL_SUCCESS){
            throw runtime_error("1D-enqueue of HC10 potential 5 kernel failed.");
        }
        
        workGridB->markReciGPUDirty();
        
        workGridB->multiplyGVectorsZ();        
        workGridB->enqueueBackwardTransform();
        
        err  = clSetKernelArg(hcP7Kernel, 0, sizeof(cl_mem), &realPotBuff);
        err |= clSetKernelArg(hcP7Kernel, 1, sizeof(cl_mem), &realGridB);
        err |= clSetKernelArg(hcP7Kernel, 2, sizeof(double), &preFactor);
        if(err != CL_SUCCESS){
            cerr << "ERROR in setting a HC10 potential 7 kernel argument: " << err << endl;
            throw runtime_error("Failed to set a HC10 potential 7 kernel argument.");
        }
        
        // enqueue the kernel over the entire "1D'd" grid
        err = clEnqueueNDRangeKernel(_queue, hcP7Kernel, 1, NULL, &enqSize, NULL, 0, NULL, NULL);
        if (err != CL_SUCCESS){
            throw runtime_error("1D-enqueue of HC10 potential 7 kernel failed.");
        }
                
        const double integral = result->integrate();
        
        workGridB.reset();
        result.reset();
        
        clReleaseMemObject(k0Buff);
        clReleaseMemObject(k1Buff);
        clReleaseMemObject(ssBuff);
        clReleaseMemObject(binnedBuff);
        clReleaseMemObject(intpotBuff);
        clReleaseMemObject(pottmpBuff);
        clReleaseMemObject(gNormsBuff);
        
        return preFactor*integral;
    }
    
    unique_ptr<StressTensor> calcStress(const CartesianOCLOOPGrid& grid) const {
        throw runtime_error("XXX implement me");
    }
    
private:
    
    const size_t _numEta;
    const double _etaStep;
    
    unique_ptr<vec> _w;
    unique_ptr<vec> _wLow;
    unique_ptr<vec> _wHigh;
    
    unique_ptr<vec> _w1;
    unique_ptr<vec> _w1Low;
    unique_ptr<vec> _w1High;
    
    // this setting relies on the fact that w/w' as handed over are defined in [0.0,50.0] with an increment of 0.001
    // empirically, we observed that the requested values are typically in [0,10]
    const size_t _KERNELSPLIT = 10000;
    
    const double _CUTXIHIGH = 1000.0;
    const double _CUTXILOW = 0.001;
    const double _CTF = 2.87123400018819;
    
    const double _alpha;
    const double _beta;
    const double _hc_lambda;
    const double _rhoS;
    const double _c;
    const double _midXi;
    const double _refRatio;
    const double _cutoffDens;
    
    cl_context _ctx;
    cl_command_queue _queue;
    
    cl_program hcOCLProg;
    
    cl_kernel hcE1Kernel;
    size_t _localhcE1Size;
    
    cl_kernel hcE2Kernel;
    size_t _localhcE2Size;
    
    cl_kernel hcE3Kernel;
    size_t _localhcE3Size;
    
    cl_kernel hcE3ZeroKernel;
    size_t _localhcE3ZeroSize;
    
    cl_kernel hcE4Kernel;
    size_t _localhcE4Size;
    
    cl_kernel hcP0Kernel;
    size_t _localhcP0Size;
    
    cl_kernel hcP1Kernel;
    size_t _localhcP1Size;
    
    cl_kernel hcP2Kernel;
    size_t _localhcP2Size;
    
    cl_kernel hcP2ZeroKernel;
    size_t _localhcP2ZeroSize;
    
    cl_kernel hcP3Kernel;
    size_t _localhcP3Size;
    
    cl_kernel hcP4Kernel;
    size_t _localhcP4Size;
    
    cl_kernel hcP5Kernel;
    size_t _localhcP5Size;
    
    cl_kernel hcP6Kernel;
    size_t _localhcP6Size;
    
    cl_kernel hcP7Kernel;
    size_t _localhcP7Size;
    
    cl_kernel binKernel;
    
    cl_kernel calcXiKernel;
    
    cl_kernel calcXiAndSSKernel;
    
    cl_kernel hc10SimpleKernel;
    
    cl_mem w0Buff;
    cl_mem w1Buff;
    
    void setupKernels(CartesianOCLOOPGrid* example){
        
        _ctx = example->getGPUContext();
        clRetainContext(_ctx);
        _queue = example->getGPUQueue();
        clRetainCommandQueue(_queue);
        cl_device_id* devices = example->getGPUDevices();
        cl_uint noDevices = example->getNoGPUDevices();

        const string st1 =                                      "\n" \
"__kernel void hcE1(  __global KEDFOCLV *temp1,                  \n" \
"    __global KEDFOCLV *temp2, __global const KEDFOCLV *xi,      \n" \
"    __global const KEDFOCLV *binned, double const incr,         \n" \
"    const double binI){                                         \n" \
"    const size_t idx = get_global_id(0);                        \n" \
"    const KEDFOCLV t = (xi[idx] - binI) / incr;                 \n" \
"    const KEDFOCLV tSq = t*t;                                   \n" \
"    const KEDFOCLV t3 = tSq*t;                                  \n" \
"    const KEDFOCLV h0 = 2*t3 - 3*tSq + 1;                       \n" \
"    const KEDFOCLV h1 = t3 - 2*tSq + t;                         \n" \
"    temp1[idx] = binned[idx] * h0;                              \n" \
"    temp2[idx] = binned[idx] * h1;                              \n" \
"}                                                               \n" \
"__kernel void hcE2(  __global KEDFOCLV *temp1,                  \n" \
"    __global KEDFOCLV *temp2, __global const KEDFOCLV *xi,      \n" \
"    __global const KEDFOCLV *binned, const double incr,         \n" \
"    const double binI){                                         \n" \
"    const size_t idx = get_global_id(0);                        \n" \
"    const KEDFOCLV t = (xi[idx] - binI) / incr;                 \n" \
"    const KEDFOCLV tSq = t*t;                                   \n" \
"    const KEDFOCLV t3 = tSq*t;                                  \n" \
"    const KEDFOCLV h0 = -2*t3 + 3*tSq;                          \n" \
"    const KEDFOCLV h1 = t3 - tSq;                               \n" \
"    temp1[idx] = binned[idx] * h0;                              \n" \
"    temp2[idx] = binned[idx] * h1;                              \n" \
"}                                                               \n" \
"__kernel void hcE3(  __global double *recResult,                \n" \
"    __global const double *k0, __global const double *tmpRec3,  \n" \
"    __global const double *k1, __global const double *tmpRec4,  \n" \
"    double incr, double binI1){                                 \n" \
"    const size_t idx = get_global_id(0);                        \n" \
"    const double pref = incr*k1[idx]/binI1;                     \n" \
"    const double re1 = k0[idx] * tmpRec3[2*idx];                \n" \
"    const double re2 = pref * tmpRec4[2*idx];                   \n" \
"    const double im1 = k0[idx] * tmpRec3[2*idx+1];              \n" \
"    const double im2 = pref * tmpRec4[2*idx+1];                 \n" \
"    recResult[2*idx] += re1+re2;                                \n" \
"    recResult[2*idx+1] += im1+im2;                              \n" \
"}                                                               \n" \
"__kernel void hcE3Zero(  __global double *recResult,            \n" \
"    __global const double *k0, __global const double *tmpRec3,  \n" \
"    __global const double *k1, __global const double *tmpRec4,  \n" \
"    double incr, double binI1){                                 \n" \
"    const size_t idx = get_global_id(0);                        \n" \
"    const double pref = incr*k1[idx]/binI1;                     \n" \
"    const double re1 = k0[idx] * tmpRec3[2*idx];                \n" \
"    const double re2 = pref * tmpRec4[2*idx];                   \n" \
"    const double im1 = k0[idx] * tmpRec3[2*idx+1];              \n" \
"    const double im2 = pref * tmpRec4[2*idx+1];                 \n" \
"    recResult[2*idx] = re1+re2;                                 \n" \
"    recResult[2*idx+1] = im1+im2;                               \n" \
"}                                                               \n" \
"__kernel void hcE4(  __global KEDFOCLV *result,                 \n" \
"    __global const KEDFOCLV *density, const double beta){       \n" \
"    const size_t idx = get_global_id(0);                        \n" \
"    const KEDFOCLV d = density[idx];                            \n" \
"    result[idx] *= KEDFPOWR(d,beta);                            \n" \
"}                                                               \n" \
"__kernel void hcP0(__global KEDFOCLV *pot,                      \n" \
"    __global KEDFOCLV *densBeta, __global const KEDFOCLV *dens, \n" \
"    const double beta){                                         \n" \
"    const size_t idx = get_global_id(0);                        \n" \
"    const KEDFOCLV d = dens[idx];                               \n" \
"    const KEDFOCLV densBM1 = KEDFPOWR(d,beta-1);                \n" \
"    const KEDFOCLV densB = densBM1 * d;                         \n" \
"    pot[idx] = densBM1;                                         \n" \
"    densBeta[idx] = densB;                                      \n" \
"}                                                               \n" \
"__kernel void hcP1( __global double *fkf0Rec,                   \n" \
"    __global double *fk1rfRec, __global const double *k0,       \n" \
"    __global const double *k1, __global const double *densB,    \n" \
"    const double binI){                                         \n" \
"    const size_t idx = get_global_id(0);                        \n" \
"    fkf0Rec[2*idx] = k0[idx]*densB[2*idx];                      \n" \
"    fkf0Rec[2*idx+1] = k0[idx]*densB[2*idx+1];                  \n" \
"    const double pref = k1[idx] / binI;                         \n" \
"    fk1rfRec[2*idx] = pref * densB[2*idx];                      \n" \
"    fk1rfRec[2*idx+1] = pref * densB[2*idx+1];                  \n" \
"}                                                               \n" \
"__kernel void hcP2( __global const double *xi,                  \n" \
"    __global double *temp1,                                     \n" \
"    __global double *temp2, __global const double *binned,      \n" \
"    __global const double *dens, __global double *intpot,       \n" \
"    __global double *pottmp, __global const double *fkf0Real,   \n" \
"    __global const double *fk1rfReal, const double binI,        \n" \
"    const double binI1, const double cutoffDens,                \n" \
"    const double incr){                                         \n" \
"    const size_t idx = get_global_id(0);                        \n" \
"    const double myXi = xi[idx];                                \n" \
"    const double t = (myXi - binI) / incr;                      \n" \
"    const double tSq = t*t;                                     \n" \
"    const double t3 = tSq*t;                                    \n" \
"    const double h0 = 2*t3 - 3*tSq + 1;                         \n" \
"    const double h1 = t3 - 2*tSq + t;                           \n" \
"    temp1[idx] = binned[idx] * h0;                              \n" \
"    temp2[idx] = binned[idx] * h1;                              \n" \
"    const double h00 = 6*tSq - 6*t;                             \n" \
"    const double h10 = 3*tSq - 4*t + 1;                         \n" \
"    const double tmp1 = h0*fkf0Real[idx]+incr*(h1*fk1rfReal[idx]); \n" \
"    const double tmp2 = (h00 * fkf0Real[idx]) / incr + h10 * fk1rfReal[idx]; \n" \
"    const double pre = (myXi < binI || myXi >= binI1 || dens[idx] < cutoffDens) ? 0.0 : 1.0;\n" \
"    intpot[idx] += pre*tmp1;                                    \n" \
"    pottmp[idx] += pre*tmp2;                                    \n" \
"}                                                               \n" \
"__kernel void hcP2Zero( __global const double *xi,              \n" \
"    __global double *temp1,                                     \n" \
"    __global double *temp2, __global const double *binned,      \n" \
"    __global const double *dens, __global double *intpot,       \n" \
"    __global double *pottmp, __global const double *fkf0Real,   \n" \
"    __global const double *fk1rfReal, const double binI,        \n" \
"    const double binI1, const double cutoffDens,                \n" \
"    const double incr){                                         \n" \
"    const size_t idx = get_global_id(0);                        \n" \
"    const double myXi = xi[idx];                                \n" \
"    const double t = (myXi - binI) / incr;                      \n" \
"    const double tSq = t*t;                                     \n" \
"    const double t3 = tSq*t;                                    \n" \
"    const double h0 = 2*t3 - 3*tSq + 1;                         \n" \
"    const double h1 = t3 - 2*tSq + t;                           \n" \
"    temp1[idx] = binned[idx] * h0;                              \n" \
"    temp2[idx] = binned[idx] * h1;                              \n" \
"    const double h00 = 6*tSq - 6*t;                             \n" \
"    const double h10 = 3*tSq - 4*t + 1;                         \n" \
"    const double tmp1 = h0*fkf0Real[idx]+incr*(h1*fk1rfReal[idx]); \n" \
"    const double tmp2 = (h00 * fkf0Real[idx]) / incr + h10 * fk1rfReal[idx]; \n" \
"    const double pre = (myXi < binI || myXi >= binI1 || dens[idx] < cutoffDens) ? 0.0 : 1.0;\n" \
"    intpot[idx] = pre*tmp1;                                     \n" \
"    pottmp[idx] = pre*tmp2;                                     \n" \
"}                                                               \n" \
"__kernel void hcP3( __global const double *xi,                  \n" \
"    __global double *temp1, __global double *temp2,             \n" \
"    __global const double *binned, __global double *dens,       \n" \
"    __global double *intpot, __global double *pottmp,           \n" \
"    __global const double *fkf0Real,                            \n" \
"    __global const double *fk1rfReal, const double binI,        \n" \
"    const double binI1, const double cutoffDens,                \n" \
"    const double incr){                                         \n" \
"    const size_t idx = get_global_id(0);                        \n" \
"    const double myXi = xi[idx];                                \n" \
"    const double t = (myXi - binI) / incr;                      \n" \
"    const double tSq = t*t;                                     \n" \
"    const double t3 = tSq*t;                                    \n" \
"    const double h0 = -2*t3 + 3*tSq;                            \n" \
"    const double h1 = t3 - tSq;                                 \n" \
"    temp1[idx] = binned[idx] * h0;                              \n" \
"    temp2[idx] = binned[idx] * h1;                              \n" \
"    const double h10 = 3*tSq - 4*t + 1;                         \n" \
"    const double h01 = -6*tSq + 6*t;                            \n" \
"    const double h11 = h10 + 2*t - 1;                           \n" \
"    const double tmp1 = h0 * fkf0Real[idx] + incr * (h1 * fk1rfReal[idx]); \n" \
"    const double tmp2 = (h01 * fkf0Real[idx])/incr + h11 * fk1rfReal[idx]; \n" \
"    const double pre = (myXi < binI || myXi >= binI1 || dens[idx] < cutoffDens) ? 0.0 : 1.0;\n" \
"    intpot[idx] += pre*tmp1;                                    \n" \
"    pottmp[idx] += pre*tmp2;                                    \n" \
"}                                                               \n" \
"__kernel void hcP4( __global KEDFOCLV *res,                     \n" \
"    __global const KEDFOCLV *dens,                              \n" \
"    __global const KEDFOCLV *densA,                             \n" \
"    __global KEDFOCLV *pot, __global KEDFOCLV *ss,              \n" \
"    __global const KEDFOCLV *pottmp,                            \n" \
"    __global const KEDFOCLV *intpot,                            \n" \
"    const double hc_lambda, const double alpha,                 \n" \
"    const double beta){                                         \n" \
"    const size_t idx = get_global_id(0);                        \n" \
"    const KEDFOCLV d = dens[idx];                               \n" \
"    const KEDFOCLV densBM1 = pot[idx];                          \n" \
"    const KEDFOCLV dA = densA[idx];                             \n" \
"    const KEDFOCLV tmp = res[idx];                              \n" \
"    const KEDFOCLV tmp2 = densBM1 * tmp;                        \n" \
"    const KEDFOCLV potTmp2 = pottmp[idx] * dA;                  \n" \
"    const KEDFOCLV kF = cbrt(3*M_PI*M_PI*d);                    \n" \
"    const KEDFOCLV pxiprho = kF/(3*d)*(1-7*hc_lambda*ss[idx]);  \n" \
"    const KEDFOCLV potPart1 = alpha*dA/d*intpot[idx];           \n" \
"    const KEDFOCLV potPart2 = beta*tmp2;                        \n" \
"    const KEDFOCLV potPart3 = pxiprho * potTmp2;                \n" \
"    const KEDFOCLV rhoSq = d*d;                                 \n" \
"    const KEDFOCLV rho4 = rhoSq*rhoSq;                          \n" \
"    const KEDFOCLV rho8 = rho4*rho4;                            \n" \
"    const KEDFOCLV rho83 = cbrt(rho8);                          \n" \
"    const KEDFOCLV t3 = (kF * hc_lambda * 2 / rho83) * potTmp2; \n" \
"    ss[idx] = t3;                                               \n" \
"    res[idx] = tmp2 * d;                                        \n" \
"    pot[idx] = potPart1 + potPart2 + potPart3;                  \n" \
"}                                                               \n" \
"__kernel void hcP5( __global KEDFOCLV *grid,                    \n" \
"    __global const KEDFOCLV *ss){                               \n" \
"    const size_t idx = get_global_id(0);                        \n" \
"    grid[idx] *= ss[idx];                                       \n" \
"}                                                               \n" \
"__kernel void hcP6( __global KEDFOCLV *pot,                     \n" \
"    __global const KEDFOCLV *grid){                             \n" \
"    const size_t idx = get_global_id(0);                        \n" \
"    pot[idx] -= grid[idx];                                      \n" \
"}                                                               \n" \
"__kernel void hcP7( __global KEDFOCLV *pot,                     \n" \
"    __global const KEDFOCLV *grid, const double preFactor){     \n" \
"    const size_t idx = get_global_id(0);                        \n" \
"    pot[idx] -= grid[idx];                                      \n" \
"    pot[idx] *= preFactor;                                      \n" \
"}                                                               \n" \
"__kernel void binner( __global const double *xi,                \n" \
"    __global const double *dens, __global double *binned,       \n" \
"    __global const double *densA,                               \n" \
"    const double cutoffDens, const double binI,                 \n" \
"    const double binI1){                                        \n" \
"    const size_t idx = get_global_id(0);                        \n" \
"    const double myXi = xi[idx];                                \n" \
"    const double mult = (myXi < binI || myXi >= binI1 || dens[idx] < cutoffDens) ? 0.0 : 1.0; \n" \
"    binned[idx] = mult*densA[idx];                              \n" \
"}                                                               \n" \
"__kernel void calcXi( __global double *xi,                      \n" \
"    __global const double *dens,                                \n" \
"    __global const double *gradSq, const double hc_lambda,      \n" \
"    const double CUTXILOW, const double CUTXIHIGH){             \n" \
"    const size_t idx = get_global_id(0);                        \n" \
"    const double rho = dens[idx];                               \n" \
"    const double rhoSq = rho*rho;                               \n" \
"    const double rho4 = rhoSq*rhoSq;                            \n" \
"    const double rho8 = rho4*rho4;                              \n" \
"    const double rho83 = cbrt(rho8);                            \n" \
"    const double ss = gradSq[idx] / rho83;                      \n" \
"    const double t = 3*M_PI*M_PI*rho;                           \n" \
"    const double t13 = cbrt(t);                                 \n" \
"    const double xiE = t13 * (1 + hc_lambda * ss);              \n" \
"    if(xiE > CUTXIHIGH){                                        \n" \
"       xi[idx] = CUTXIHIGH;                                     \n" \
"    } else if(xiE < CUTXILOW){                                  \n" \
"       xi[idx] = CUTXILOW;                                      \n" \
"    } else {                                                    \n" \
"       xi[idx] = xiE;                                           \n" \
"    }                                                           \n" \
"}                                                               \n" \
"__kernel void calcXiAndSS( __global double *xi,                 \n" \
"    __global double *ss, __global const double *dens,           \n" \
"    __global const double *gradSq, const double hc_lambda,      \n" \
"    const double CUTXILOW, const double CUTXIHIGH){             \n" \
"    const size_t idx = get_global_id(0);                        \n" \
"    const double rho = dens[idx];                               \n" \
"    const double rhoSq = rho*rho;                               \n" \
"    const double rho4 = rhoSq*rhoSq;                            \n" \
"    const double rho8 = rho4*rho4;                              \n" \
"    const double rho83 = cbrt(rho8);                            \n" \
"    const double ssX = gradSq[idx] / rho83;                     \n" \
"    ss[idx] = ssX;                                              \n" \
"    const double t = 3*M_PI*M_PI*rho;                           \n" \
"    const double t13 = cbrt(t);                                 \n" \
"    const double xiE = t13 * (1 + hc_lambda * ssX);             \n" \
"    if(xiE > CUTXIHIGH){                                        \n" \
"       xi[idx] = CUTXIHIGH;                                     \n" \
"    } else if(xiE < CUTXILOW){                                  \n" \
"       xi[idx] = CUTXILOW;                                      \n" \
"    } else {                                                    \n" \
"       xi[idx] = xiE;                                           \n" \
"    }                                                           \n" \
"}                                                               \n" \
"__kernel void hc10KernelsSimple(__global double *k0, __global double *k1, \n" \
"    __global const double *gNorms, __global const double *w,    \n" \
"    __global const double *w1, const double xi,                 \n" \
"    const double xi3, const double t8xi3, const double etaStep, \n" \
"    const int numEta){                                          \n" \
"    const size_t idx = get_global_id(0);                        \n" \
"    const double eta = gNorms[idx] / (2 * xi);                  \n" \
"    const size_t ind = floor(eta/etaStep);                      \n" \
"    double tmp;                                                 \n" \
"    double tmp1;                                                \n" \
"    if(ind >= numEta-1){                                        \n" \
"       tmp = w[numEta-1];                                       \n" \
"       tmp1 = w1[numEta-1];                                     \n" \
"    } else {                                                    \n" \
"       tmp = (w[ind] + w[ind+1]) * 0.5;                         \n" \
"       tmp1 = (w1[ind] + w1[ind+1]) * 0.5;                      \n" \
"    }                                                           \n" \
"    k0[idx] = tmp / t8xi3;                                      \n" \
"    k1[idx] = (-tmp1-3*tmp)/ t8xi3;                             \n" \
"}                                                               \n" \
                                                                "\n" ;        
        
        const string macros = example->getMacroDefinitions();
    
        ostringstream os;
        os << macros;
        os << st1;

        const string s = os.str();
        const char* st = s.c_str();
        
        cl_int err;
        this->hcOCLProg = clCreateProgramWithSource(_ctx, 1, (const char**) &st, NULL, &err);
        if(!hcOCLProg || err != CL_SUCCESS){
            cerr << "ERROR to create HC OCL program " << err << endl;
            throw runtime_error("Failed to create HC OCL program.");
        }
        err = clBuildProgram(hcOCLProg, noDevices, devices, example->getCompilationOptions(), NULL, NULL);
        if(err != CL_SUCCESS){
            cerr << "ERROR in building HC OCL program " << err << endl;
            
            cl_build_status status;
            // check build error and build status first
            clGetProgramBuildInfo(hcOCLProg, devices[0], CL_PROGRAM_BUILD_STATUS, 
                sizeof(cl_build_status), &status, NULL);
 
            if(!(err == CL_BUILD_PROGRAM_FAILURE && status == 0)){ // this is what I observe currently on NVIDIA and it seems to be caused by the KEDFOCL macro, makes no sense
            
                // check build log
                size_t logSize;
                clGetProgramBuildInfo(hcOCLProg, devices[0], 
                        CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);
                char* programLog = (char*) calloc (logSize+1, sizeof(char));
                clGetProgramBuildInfo(hcOCLProg, devices[0], 
                        CL_PROGRAM_BUILD_LOG, logSize+1, programLog, NULL);
                printf("Build failed; error=%d, status=%d, programLog:nn%s \n", 
                        err, status, programLog);
                free(programLog);
        
                throw runtime_error("Could not build HC OCL program.");
            }
        }
        
        this->hcE1Kernel = clCreateKernel(hcOCLProg, "hcE1", &err);
        if(err != CL_SUCCESS){
            cerr << "ERROR in creating HC 1st energy kernel " << err << endl;
            throw runtime_error("Could not create HC 1st energy kernel.");
        }
        
        err = clGetKernelWorkGroupInfo(hcE1Kernel, devices[0], CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &_localhcE1Size, NULL);
        if (err != CL_SUCCESS){
            cerr << "ERROR to inquire local work group size for HC 1st energy kernel " << err << endl;
            throw runtime_error("Not possible to inquire local work group size of HC 1st energy kernel.");
        }
        
        this->hcE2Kernel = clCreateKernel(hcOCLProg, "hcE2", &err);
        if(err != CL_SUCCESS){
            cerr << "ERROR in creating HC 2nd energy kernel " << err << endl;
            throw runtime_error("Could not create HC 2nd energy kernel.");
        }
        
        err = clGetKernelWorkGroupInfo(hcE2Kernel, devices[0], CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &_localhcE2Size, NULL);
        if (err != CL_SUCCESS){
            cerr << "ERROR to inquire local work group size for HC 2nd energy kernel " << err << endl;
            throw runtime_error("Not possible to inquire local work group size of HC 2nd energy kernel.");
        }
        
        this->hcE3Kernel = clCreateKernel(hcOCLProg, "hcE3", &err);
        if(err != CL_SUCCESS){
            cerr << "ERROR in creating HC 3rd energy kernel " << err << endl;
            throw runtime_error("Could not create HC 3rd energy kernel.");
        }
        
        err = clGetKernelWorkGroupInfo(hcE3Kernel, devices[0], CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &_localhcE3Size, NULL);
        if (err != CL_SUCCESS){
            cerr << "ERROR to inquire local work group size for HC 3rd energy kernel " << err << endl;
            throw runtime_error("Not possible to inquire local work group size of HC 3rd energy kernel.");
        }
        
        this->hcE3ZeroKernel = clCreateKernel(hcOCLProg, "hcE3Zero", &err);
        if(err != CL_SUCCESS){
            cerr << "ERROR in creating HC 3rd energy (zero) kernel " << err << endl;
            throw runtime_error("Could not create HC 3rd energy (zero) kernel.");
        }
        
        err = clGetKernelWorkGroupInfo(hcE3ZeroKernel, devices[0], CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &_localhcE3ZeroSize, NULL);
        if (err != CL_SUCCESS){
            cerr << "ERROR to inquire local work group size for HC 3rd energy (zero) kernel " << err << endl;
            throw runtime_error("Not possible to inquire local work group size of HC 3rd energy (zero) kernel.");
        }
        
        this->hcE4Kernel = clCreateKernel(hcOCLProg, "hcE4", &err);
        if(err != CL_SUCCESS){
            cerr << "ERROR in creating HC 4th energy kernel " << err << endl;
            throw runtime_error("Could not create HC 4th energy kernel.");
        }
        
        err = clGetKernelWorkGroupInfo(hcE4Kernel, devices[0], CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &_localhcE4Size, NULL);
        if (err != CL_SUCCESS){
            cerr << "ERROR to inquire local work group size for HC 4th energy kernel " << err << endl;
            throw runtime_error("Not possible to inquire local work group size of HC 4th energy kernel.");
        }
        
        this->hcP0Kernel = clCreateKernel(hcOCLProg, "hcP0", &err);
        if(err != CL_SUCCESS){
            cerr << "ERROR in creating HC 0th potential kernel " << err << endl;
            throw runtime_error("Could not create HC 0th potential kernel.");
        }
        
        err = clGetKernelWorkGroupInfo(hcP0Kernel, devices[0], CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &_localhcP0Size, NULL);
        if (err != CL_SUCCESS){
            cerr << "ERROR to inquire local work group size for HC 0th potential kernel " << err << endl;
            throw runtime_error("Not possible to inquire local work group size of HC 0th potential kernel.");
        }
        
        this->hcP1Kernel = clCreateKernel(hcOCLProg, "hcP1", &err);
        if(err != CL_SUCCESS){
            cerr << "ERROR in creating HC 1st potential kernel " << err << endl;
            throw runtime_error("Could not create HC 1st potential kernel.");
        }
        
        err = clGetKernelWorkGroupInfo(hcP1Kernel, devices[0], CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &_localhcP1Size, NULL);
        if (err != CL_SUCCESS){
            cerr << "ERROR to inquire local work group size for HC 1st potential kernel " << err << endl;
            throw runtime_error("Not possible to inquire local work group size of HC 1st potential kernel.");
        }
        
        this->hcP2Kernel = clCreateKernel(hcOCLProg, "hcP2", &err);
        if(err != CL_SUCCESS){
            cerr << "ERROR in creating HC 2nd potential kernel " << err << endl;
            throw runtime_error("Could not create HC 2nd potential kernel.");
        }
        
        err = clGetKernelWorkGroupInfo(hcP2Kernel, devices[0], CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &_localhcP2Size, NULL);
        if (err != CL_SUCCESS){
            cerr << "ERROR to inquire local work group size for HC 2nd potential kernel " << err << endl;
            throw runtime_error("Not possible to inquire local work group size of HC 2nd potential kernel.");
        }
        
        this->hcP2ZeroKernel = clCreateKernel(hcOCLProg, "hcP2Zero", &err);
        if(err != CL_SUCCESS){
            cerr << "ERROR in creating HC 2nd potential (zero) kernel " << err << endl;
            throw runtime_error("Could not create HC 2nd potential (zero) kernel.");
        }
        
        err = clGetKernelWorkGroupInfo(hcP2ZeroKernel, devices[0], CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &_localhcP2ZeroSize, NULL);
        if (err != CL_SUCCESS){
            cerr << "ERROR to inquire local work group size for HC 2nd potential (zero) kernel " << err << endl;
            throw runtime_error("Not possible to inquire local work group size of HC 2nd potential (zero) kernel.");
        }
        
        this->hcP3Kernel = clCreateKernel(hcOCLProg, "hcP3", &err);
        if(err != CL_SUCCESS){
            cerr << "ERROR in creating HC 3rd potential kernel " << err << endl;
            throw runtime_error("Could not create HC 3rd potential kernel.");
        }
        
        err = clGetKernelWorkGroupInfo(hcP3Kernel, devices[0], CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &_localhcP3Size, NULL);
        if (err != CL_SUCCESS){
            cerr << "ERROR to inquire local work group size for HC 3rd potential kernel " << err << endl;
            throw runtime_error("Not possible to inquire local work group size of HC 3rd potential kernel.");
        }
        
        this->hcP4Kernel = clCreateKernel(hcOCLProg, "hcP4", &err);
        if(err != CL_SUCCESS){
            cerr << "ERROR in creating HC 4th potential kernel " << err << endl;
            throw runtime_error("Could not create HC 4th potential kernel.");
        }
        
        err = clGetKernelWorkGroupInfo(hcP4Kernel, devices[0], CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &_localhcP4Size, NULL);
        if (err != CL_SUCCESS){
            cerr << "ERROR to inquire local work group size for HC 4th potential kernel " << err << endl;
            throw runtime_error("Not possible to inquire local work group size of HC 4th potential kernel.");
        }
        
        this->hcP5Kernel = clCreateKernel(hcOCLProg, "hcP5", &err);
        if(err != CL_SUCCESS){
            cerr << "ERROR in creating HC 5th potential kernel " << err << endl;
            throw runtime_error("Could not create HC 5th potential kernel.");
        }
        
        err = clGetKernelWorkGroupInfo(hcP5Kernel, devices[0], CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &_localhcP5Size, NULL);
        if (err != CL_SUCCESS){
            cerr << "ERROR to inquire local work group size for HC 5th potential kernel " << err << endl;
            throw runtime_error("Not possible to inquire local work group size of HC 5th potential kernel.");
        }
        
        this->hcP6Kernel = clCreateKernel(hcOCLProg, "hcP6", &err);
        if(err != CL_SUCCESS){
            cerr << "ERROR in creating HC 6th potential kernel " << err << endl;
            throw runtime_error("Could not create HC 6th potential kernel.");
        }
        
        err = clGetKernelWorkGroupInfo(hcP6Kernel, devices[0], CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &_localhcP6Size, NULL);
        if (err != CL_SUCCESS){
            cerr << "ERROR to inquire local work group size for HC 6th potential kernel " << err << endl;
            throw runtime_error("Not possible to inquire local work group size of HC 6th potential kernel.");
        }
        
        this->hcP7Kernel = clCreateKernel(hcOCLProg, "hcP7", &err);
        if(err != CL_SUCCESS){
            cerr << "ERROR in creating HC 7th potential kernel " << err << endl;
            throw runtime_error("Could not create HC 7th potential kernel.");
        }
        
        err = clGetKernelWorkGroupInfo(hcP7Kernel, devices[0], CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &_localhcP7Size, NULL);
        if (err != CL_SUCCESS){
            cerr << "ERROR to inquire local work group size for HC 7th potential kernel " << err << endl;
            throw runtime_error("Not possible to inquire local work group size of HC 7th potential kernel.");
        }
        
        this->binKernel = clCreateKernel(hcOCLProg, "binner", &err);
        if(err != CL_SUCCESS){
            cerr << "ERROR in creating binning kernel " << err << endl;
            throw runtime_error("Could not create binning kernel.");
        }
        
        this->calcXiKernel = clCreateKernel(hcOCLProg, "calcXi", &err);
        if(err != CL_SUCCESS){
            cerr << "ERROR in creating xi kernel " << err << endl;
            throw runtime_error("Could not create xi kernel.");
        }
        
        this->calcXiAndSSKernel = clCreateKernel(hcOCLProg, "calcXiAndSS", &err);
        if(err != CL_SUCCESS){
            cerr << "ERROR in creating xi and ss kernel " << err << endl;
            throw runtime_error("Could not create xi and ss kernel.");
        }
        
        this->hc10SimpleKernel = clCreateKernel(hcOCLProg, "hc10KernelsSimple", &err);
        if(err != CL_SUCCESS){
            cerr << "ERROR in creating simple HC10 assembly kernel " << err << endl;
            throw runtime_error("Could not create simple HC10 assembly kernel.");
        }
    }
    
    void calcXiGPU(CartesianOCLOOPGrid* density, cl_mem *xi, const size_t totalSize) const {
        
        unique_ptr<CartesianOCLOOPGrid> gradientSquared(density->gradientSquared());
        const cl_mem gradSq = gradientSquared->readRealGPUBuffer();
        
        const cl_mem dens = density->readRealGPUBuffer();
        
        cl_int err;
        err  = clSetKernelArg(calcXiKernel, 0, sizeof(cl_mem), xi);
        err |= clSetKernelArg(calcXiKernel, 1, sizeof(cl_mem), &dens);
        err |= clSetKernelArg(calcXiKernel, 2, sizeof(cl_mem), &gradSq);
        err |= clSetKernelArg(calcXiKernel, 3, sizeof(double), &_hc_lambda);
        err |= clSetKernelArg(calcXiKernel, 4, sizeof(double), &_CUTXILOW);
        err |= clSetKernelArg(calcXiKernel, 5, sizeof(double), &_CUTXIHIGH);
        if(err != CL_SUCCESS){
            cerr << "ERROR in setting a calc XI kernel argument: " << err << endl;
            throw runtime_error("Failed to set a calc XI kernel argument.");
        }
            
        // enqueue the kernel over the entire "1D'd" grid
        err = clEnqueueNDRangeKernel(_queue, calcXiKernel, 1, NULL, &totalSize, NULL, 0, NULL, NULL);
        if (err != CL_SUCCESS){
            throw runtime_error("1D-enqueue of calc XI kernel failed.");
        }
        
        gradientSquared.reset();
    }
    
    void calcXiAndSSGPU(CartesianOCLOOPGrid* density, cl_mem *xi, cl_mem *ss, const size_t totalSize) const {
        
        unique_ptr<CartesianOCLOOPGrid> gradientSquared(density->gradientSquared());
        const cl_mem gradSq = gradientSquared->readRealGPUBuffer();
        
        const cl_mem dens = density->readRealGPUBuffer();
        
        cl_int err;
        err  = clSetKernelArg(calcXiAndSSKernel, 0, sizeof(cl_mem), xi);
        err |= clSetKernelArg(calcXiAndSSKernel, 1, sizeof(cl_mem), ss);
        err |= clSetKernelArg(calcXiAndSSKernel, 2, sizeof(cl_mem), &dens);
        err |= clSetKernelArg(calcXiAndSSKernel, 3, sizeof(cl_mem), &gradSq);
        err |= clSetKernelArg(calcXiAndSSKernel, 4, sizeof(double), &_hc_lambda);
        err |= clSetKernelArg(calcXiAndSSKernel, 5, sizeof(double), &_CUTXILOW);
        err |= clSetKernelArg(calcXiAndSSKernel, 6, sizeof(double), &_CUTXIHIGH);
        if(err != CL_SUCCESS){
            cerr << "ERROR in setting a calc XI and SS kernel argument: " << err << endl;
            throw runtime_error("Failed to set a calc XI and SS kernel argument.");
        }
            
        // enqueue the kernel over the entire "1D'd" grid
        err = clEnqueueNDRangeKernel(_queue, calcXiAndSSKernel, 1, NULL, &totalSize, NULL, 0, NULL, NULL);
        if (err != CL_SUCCESS){
            throw runtime_error("1D-enqueue of calc XI and SS kernel failed.");
        }
        
        gradientSquared.reset();
    }
    
    // XXX important: I think we can solve the performance impact of this almost completely
    // 1) going to two lookups (ind is typically below 10,000)
    // 2) most of the features are >10 -> keep this dense, make the other one sparser
    // 3) doing *actually* linear interpolation between points
    // 4) maybe 3) will allow to go more sparse for *both* tables
    // so, w/o 4) we would have 1-10 at 10,000 points -> 80k
    //            and 10-50 at, say, 20,000 points -> 160k
    // if 4) allows us a factor 2-5 reduction -> 40-16k for the smaller one
    //            and 80-32k for the bigger one.
    void calcHC10Kernels(const double xi, cube* k0, cube* k1, const cube* gNorms) const {
        
        const size_t nSlicesRec = k0->n_slices;
        const size_t nRowsRec = k0->n_rows;
        const size_t nColsRec = k0->n_cols;
        
        const double xi3 = xi*xi*xi;
        const double t8xi3 = 8*xi3;
        
        #pragma omp parallel for default(none) shared(k0, k1, gNorms)
        for(size_t x = 0; x < nSlicesRec; ++x){
            for(size_t col = 0; col < nColsRec; ++col){
                for(size_t row = 0; row < nRowsRec; ++row){
                    
                    const double eta = gNorms->at(row,col,x) / (2 * xi);
                    const size_t ind = floor(eta/_etaStep);
                    
                    // check if our index is within our lookup table
                    double tmp;
                    double tmp1;
                    if(ind >= _numEta-1){
                        tmp = _w->at(_numEta-1);
                        tmp1 = _w1->at(_numEta-1);
#ifndef LIBKEDF_ILIKESLOWCODE
                    } else if(ind < _KERNELSPLIT-1){
                        tmp = (_wLow->at(ind) + _wLow->at(ind + 1)) * 0.5;
                        tmp1 = (_w1Low->at(ind) + _w1Low->at(ind + 1) ) * 0.5;
                    } else {
                        const size_t ind2 = ind - _KERNELSPLIT + 1; // we are shifting one up
                        tmp = (_wHigh->at(ind2) + _wHigh->at(ind2 + 1)) * 0.5;
                        tmp1 = (_w1High->at(ind2) + _w1High->at(ind2 + 1) ) * 0.5;
                    }
                    
#else
                    } else {
                        tmp = (_w->at(ind) + _w->at(ind + 1)) * 0.5;
                        tmp1 = (_w1->at(ind) + _w1->at(ind + 1) ) * 0.5;
                    }
#endif
                    
                    k0->at(row,col,x) = tmp / t8xi3;
                    k1->at(row,col,x) = (-tmp1-3*tmp)/ t8xi3;
                }
            }
        }
    }
    
};

#endif /* HUANGCARTEROCL_HPP */

