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

#ifndef TAYLOREDWANGGOVINDCARTEROCL_HPP
#define TAYLOREDWANGGOVINDCARTEROCL_HPP

#include <string>
#include "CartesianOCLOOPGrid.hpp"

template<>
class TayloredWangGovindCarter<CartesianOCLOOPGrid>: public KEDF<CartesianOCLOOPGrid> {
    
public:
    TayloredWangGovindCarter(CartesianOCLOOPGrid* example, const double alpha, const double beta, const double gamma, const double rhoS, cube* kernel0th,
            cube* kernel1st) : _alpha(alpha), _beta(beta), _rhoS(rhoS), _sndOrder(false), _vacCutoff(false),
                    _kernel0th(kernel0th), _kernel1st(kernel1st), _kernel2nd(NULL), _kernel3rd(NULL) {
        // XXX make vacCutoff configurable

        // setup internal kernels
        setupKernels(example, false);
        
        // push the kernels to the GPU where they belong
        const size_t totReciSize = _kernel0th->n_rows * _kernel0th->n_slices * _kernel0th->n_cols;
        const size_t halfReciMem = totReciSize*sizeof(double);
        
        cl_int err;
        _keGPUMem0th = clCreateBuffer(_ctx, CL_MEM_READ_ONLY, halfReciMem, NULL, &err);
        if(err != CL_SUCCESS){
            cerr << "ERROR in allocating GPU memory of size " << halfReciMem << " with error " << err << endl;
            throw runtime_error("Failed to allocate GPU memory");
        }
        _keGPUMem1st = clCreateBuffer(_ctx, CL_MEM_READ_ONLY, halfReciMem, NULL, &err);
        if(err != CL_SUCCESS){
            cerr << "ERROR in allocating GPU memory of size " << halfReciMem << " with error " << err << endl;
            throw runtime_error("Failed to allocate GPU memory");
        }
        
        err = clEnqueueWriteBuffer(_queue, _keGPUMem0th, CL_FALSE, 0, halfReciMem, _kernel0th->memptr(), 0, NULL, NULL );
        if(err != CL_SUCCESS){
            cerr << "ERROR to write 0th kernel data to GPU " << err << endl;
            throw runtime_error("Failed to write kernel data to GPU.");
        }
        err = clEnqueueWriteBuffer(_queue, _keGPUMem1st, CL_FALSE, 0, halfReciMem, _kernel1st->memptr(), 0, NULL, NULL );
        if(err != CL_SUCCESS){
            cerr << "ERROR to write 1st kernel data to GPU " << err << endl;
            throw runtime_error("Failed to write kernel data to GPU.");
        }        
    }

    TayloredWangGovindCarter(CartesianOCLOOPGrid* example, const double alpha, const double beta, const double gamma, const double rhoS, cube* kernel0th,
            cube* kernel1st, cube* kernel2nd, cube* kernel3rd)
            : _alpha(alpha), _beta(beta), _rhoS(rhoS), _sndOrder(true), _vacCutoff(false),
            _kernel0th(kernel0th), _kernel1st(kernel1st), _kernel2nd(kernel2nd), _kernel3rd(kernel3rd){
        // XXX make vacCutoff configurable
                
        // setup internal kernels
        setupKernels(example, true);
        
        // push the kernels to the GPU where they belong
        const size_t totReciSize = _kernel0th->n_rows * _kernel0th->n_slices * _kernel0th->n_cols;
        const size_t halfReciMem = totReciSize*sizeof(double);
        
        cl_int err;
        _keGPUMem0th = clCreateBuffer(_ctx, CL_MEM_READ_ONLY, halfReciMem, NULL, &err);
        if(err != CL_SUCCESS){
            cerr << "ERROR in allocating GPU memory of size " << halfReciMem << " with error " << err << endl;
            throw runtime_error("Failed to allocate GPU memory");
        }
        _keGPUMem1st = clCreateBuffer(_ctx, CL_MEM_READ_ONLY, halfReciMem, NULL, &err);
        if(err != CL_SUCCESS){
            cerr << "ERROR in allocating GPU memory of size " << halfReciMem << " with error " << err << endl;
            throw runtime_error("Failed to allocate GPU memory");
        }
        _keGPUMem2nd = clCreateBuffer(_ctx, CL_MEM_READ_ONLY, halfReciMem, NULL, &err);
        if(err != CL_SUCCESS){
            cerr << "ERROR in allocating GPU memory of size " << halfReciMem << " with error " << err << endl;
            throw runtime_error("Failed to allocate GPU memory");
        }
        _keGPUMem3rd = clCreateBuffer(_ctx, CL_MEM_READ_ONLY, halfReciMem, NULL, &err);
        if(err != CL_SUCCESS){
            cerr << "ERROR in allocating GPU memory of size " << halfReciMem << " with error " << err << endl;
            throw runtime_error("Failed to allocate GPU memory");
        }
        
        err = clEnqueueWriteBuffer(_queue, _keGPUMem0th, CL_FALSE, 0, halfReciMem, _kernel0th->memptr(), 0, NULL, NULL );
        if(err != CL_SUCCESS){
            cerr << "ERROR to write 0th kernel data to GPU " << err << endl;
            throw runtime_error("Failed to write kernel data to GPU.");
        }
        err = clEnqueueWriteBuffer(_queue, _keGPUMem1st, CL_FALSE, 0, halfReciMem, _kernel1st->memptr(), 0, NULL, NULL );
        if(err != CL_SUCCESS){
            cerr << "ERROR to write 1st kernel data to GPU " << err << endl;
            throw runtime_error("Failed to write kernel data to GPU.");
        }
        err = clEnqueueWriteBuffer(_queue, _keGPUMem2nd, CL_FALSE, 0, halfReciMem, _kernel2nd->memptr(), 0, NULL, NULL );
        if(err != CL_SUCCESS){
            cerr << "ERROR to write 2nd kernel data to GPU " << err << endl;
            throw runtime_error("Failed to write kernel data to GPU.");
        }
        err = clEnqueueWriteBuffer(_queue, _keGPUMem3rd, CL_FALSE, 0, halfReciMem, _kernel3rd->memptr(), 0, NULL, NULL );
        if(err != CL_SUCCESS){
            cerr << "ERROR to write 3rd kernel data to GPU " << err << endl;
            throw runtime_error("Failed to write kernel data to GPU.");
        }
    }

    ~TayloredWangGovindCarter(){
        clReleaseCommandQueue(_queue);
        clReleaseContext(_ctx);
        clReleaseMemObject(_keGPUMem0th);
        clReleaseMemObject(_keGPUMem1st);
        if(_sndOrder){
            clReleaseMemObject(_keGPUMem2nd);
            clReleaseMemObject(_keGPUMem3rd);
        }
        
        clReleaseProgram(wgcOCLProg);
        if(!_sndOrder){
            clReleaseKernel(wgc1stE1Kernel);
            clReleaseKernel(wgc1stE2Kernel);
            clReleaseKernel(wgc1stE3Kernel);
            clReleaseKernel(wgc1stP1Kernel);
            clReleaseKernel(wgc1stP2Kernel);
        } else {
            clReleaseKernel(wgc2ndE1Kernel);
            clReleaseKernel(wgc2ndE2Kernel);
            clReleaseKernel(wgc2ndE3Kernel);
            clReleaseKernel(wgc2ndP1Kernel);
            clReleaseKernel(wgc2ndP2Kernel);
            clReleaseKernel(wgc2ndP3Kernel);
        }
    }

    string getMethodDescription() const {
        if(_sndOrder){
            return "2nd order Taylor-expanded Wang-Govind-Carter (1999) KEDF) (OCL version)" ;
        } else {
            return "1st order Taylor-expanded Wang-Govind-Carter (1999) KEDF) (OCL version)" ;
        }
    }

    vector<string> getCitations() const {
        vector<string> citations(0);
        citations.push_back("sorry, not yet");
    
        return citations;
    }

    vector<string> getWorkingEquations() const {
        vector<string> citations(0);
        citations.push_back("sorry, not yet");
    
        return citations;
    }

    double calcEnergy(const CartesianOCLOOPGrid& grid) const {
    
        if(!_sndOrder && !_vacCutoff){
            // first order Taylor expansion, no vacuum cutoff employed
        
            unique_ptr<CartesianOCLOOPGrid> dens = grid.duplicate();
            dens->transferRealToGPU();
            
            unique_ptr<CartesianOCLOOPGrid> workGridA = grid.emptyDuplicate();
            unique_ptr<CartesianOCLOOPGrid> scr1 = grid.emptyDuplicate();
            unique_ptr<CartesianOCLOOPGrid> scr2 = grid.emptyDuplicate();
        
            const size_t totRealSize = workGridA->getTotalGridPoints();
            const size_t enqSize = totRealSize/grid.getVectortypeAlignment();
            const size_t totReciSize = _kernel0th->n_rows * _kernel0th->n_slices * _kernel0th->n_cols;
            
            const cl_mem densReal = dens->getRealGPUBuffer();
            cl_mem densityAReal = workGridA->getRealGPUBuffer();
            cl_mem scr1Real = scr1->getRealGPUBuffer();
            cl_mem scr2Real = scr2->getRealGPUBuffer();
            
            cl_mem densityAReci = workGridA->getReciGPUBuffer();
            cl_mem scr1Reci = scr1->getReciGPUBuffer();
            cl_mem scr2Reci = scr2->getReciGPUBuffer();
            
            cl_int err;
            /*
             * First WGC 1st order energy kernel
             */
            workGridA->markRealGPUClean();
            workGridA->markReciGPUDirty();
            scr1->markRealGPUClean();
            scr1->markReciGPUDirty();
            
            err  = clSetKernelArg(wgc1stE1Kernel, 0, sizeof(cl_mem), &densReal);
            err |= clSetKernelArg(wgc1stE1Kernel, 1, sizeof(cl_mem), &densityAReal);
            err |= clSetKernelArg(wgc1stE1Kernel, 2, sizeof(cl_mem), &scr1Real);
            err |= clSetKernelArg(wgc1stE1Kernel, 3, sizeof(double), &_alpha);
            err |= clSetKernelArg(wgc1stE1Kernel, 4, sizeof(double), &_rhoS);
            if(err != CL_SUCCESS){
                cerr << "ERROR in setting a WGC 1st E kernel argument: " << err << endl;
                throw runtime_error("Failed to set a WGC 1st E kernel argument.");
            }
        
            // enqueue the kernel over the entire "1D'd" grid
            err = clEnqueueNDRangeKernel(_queue, wgc1stE1Kernel, 1, NULL, &enqSize, NULL, 0, NULL, NULL);
            if (err != CL_SUCCESS){
                throw runtime_error("1D-enqueue of WGC 1st E kernel failed.");
            }
                        
            /*
             * move densA and scr1 into g space, scr2 is whatever
             */
            workGridA->enqueueForwardTransform();
            scr1->enqueueForwardTransform();
            
            /*
             * multiply with the WGC kernel expansion
             */
            scr1->markReciGPUClean();
            scr1->markRealGPUDirty();
            scr2->markReciGPUClean();
            scr2->markRealGPUDirty();
            
            err  = clSetKernelArg(wgc1stE2Kernel, 0, sizeof(cl_mem), &scr1Reci);
            err |= clSetKernelArg(wgc1stE2Kernel, 1, sizeof(cl_mem), &scr2Reci);
            err |= clSetKernelArg(wgc1stE2Kernel, 2, sizeof(cl_mem), &_keGPUMem0th);
            err |= clSetKernelArg(wgc1stE2Kernel, 3, sizeof(cl_mem), &densityAReci);
            err |= clSetKernelArg(wgc1stE2Kernel, 4, sizeof(cl_mem), &_keGPUMem1st);
            if(err != CL_SUCCESS){
                cerr << "ERROR in setting a WGC 2nd E kernel argument: " << err << endl;
                throw runtime_error("Failed to set a WGC 2nd E kernel argument.");
            }
        
            // enqueue the kernel over the entire "1D'd" grid
            err = clEnqueueNDRangeKernel(_queue, wgc1stE2Kernel, 1, NULL, &totReciSize, NULL, 0, NULL, NULL);
            if (err != CL_SUCCESS){
                throw runtime_error("1D-enqueue of WGC 2nd E kernel failed.");
            }
        
            /*
             * get the scr1 and scr2 into real space, we only need one of them to be writable
             */
            scr1->enqueueBackwardTransform();
            scr2->enqueueBackwardTransform();
            
            /*
             * the last kernel for 1st order WGC 
             */
            err  = clSetKernelArg(wgc1stE3Kernel, 0, sizeof(cl_mem), &densReal);
            err |= clSetKernelArg(wgc1stE3Kernel, 1, sizeof(cl_mem), &scr1Real);
            err |= clSetKernelArg(wgc1stE3Kernel, 2, sizeof(cl_mem), &scr2Real);
            err |= clSetKernelArg(wgc1stE3Kernel, 3, sizeof(double), &_beta);
            if(err != CL_SUCCESS){
                cerr << "ERROR in setting a WGC 3rd E kernel argument: " << err << endl;
                throw runtime_error("Failed to set a WGC 3rd E kernel argument.");
            }
        
            // enqueue the kernel over the entire "1D'd" grid
            err = clEnqueueNDRangeKernel(_queue, wgc1stE3Kernel, 1, NULL, &enqSize, NULL, 0, NULL, NULL);
            if (err != CL_SUCCESS){
                throw runtime_error("1D-enqueue of WGC 3rd E kernel failed.");
            }
            
            const double energy = _CTF*scr2->integrate();
        
            scr1.reset();
            scr2.reset();
            workGridA.reset();
            dens.reset();
            
            return energy;
        } else if(_sndOrder && !_vacCutoff){
        
            // second order Tayler expansion, no vacuum cutoff
        
            unique_ptr<CartesianOCLOOPGrid> density = grid.duplicate();
            unique_ptr<CartesianOCLOOPGrid> workGridA = grid.duplicate();
            density->transferRealToGPU();
            workGridA->transferRealToGPU();
            
            unique_ptr<CartesianOCLOOPGrid> scr1 = grid.emptyDuplicate();
            unique_ptr<CartesianOCLOOPGrid> scr2 = grid.emptyDuplicate();
            unique_ptr<CartesianOCLOOPGrid> theta = grid.emptyDuplicate();
        
            const size_t totRealSize = workGridA->getTotalGridPoints();
            const size_t enqSize = totRealSize/grid.getVectortypeAlignment();
            const size_t totReciSize = _kernel0th->n_rows * _kernel0th->n_slices * _kernel0th->n_cols;
            
            const cl_mem densReal = density->getRealGPUBuffer();
            cl_mem densityAReal = workGridA->getRealGPUBuffer();
            cl_mem scr1Real = scr1->getRealGPUBuffer();
            cl_mem scr2Real = scr2->getRealGPUBuffer();
            cl_mem thetaReal = theta->getRealGPUBuffer();
            
            cl_mem densityAReci = workGridA->getReciGPUBuffer();
            cl_mem scr1Reci = scr1->getReciGPUBuffer();
            cl_mem scr2Reci = scr2->getReciGPUBuffer();
            
            cl_int err;
            
            /*
             * assemble the first parts of the WGC expression
             */
            workGridA->markRealGPUClean();
            workGridA->markReciGPUDirty();
            scr1->markRealGPUClean();
            scr1->markReciGPUDirty();
            scr2->markRealGPUClean();
            scr2->markReciGPUDirty();
            theta->markRealGPUClean();
            theta->markReciGPUDirty();
            
            err  = clSetKernelArg(wgc2ndE1Kernel, 0, sizeof(cl_mem), &densityAReal);
            err |= clSetKernelArg(wgc2ndE1Kernel, 1, sizeof(cl_mem), &scr1Real);
            err |= clSetKernelArg(wgc2ndE1Kernel, 2, sizeof(cl_mem), &scr2Real);
            err |= clSetKernelArg(wgc2ndE1Kernel, 3, sizeof(cl_mem), &thetaReal);
            err |= clSetKernelArg(wgc2ndE1Kernel, 4, sizeof(double), &_alpha);
            err |= clSetKernelArg(wgc2ndE1Kernel, 5, sizeof(double), &_rhoS);
            if(err != CL_SUCCESS){
                cerr << "ERROR in setting a WGC 1st E kernel argument: " << err << endl;
                throw runtime_error("Failed to set a WGC 1st E kernel argument.");
            }
        
            // enqueue the kernel over the entire "1D'd" grid
            err = clEnqueueNDRangeKernel(_queue, wgc2ndE1Kernel, 1, NULL, &enqSize, NULL, 0, NULL, NULL);
            if (err != CL_SUCCESS){
                throw runtime_error("1D-enqueue of WGC 2nd E kernel failed.");
            }

            /*
             * move densA, scr1, and scr2 into g space
             */
            workGridA->enqueueForwardTransform();
            scr1->enqueueForwardTransform();
            scr2->enqueueForwardTransform();
                
            unique_ptr<CartesianOCLOOPGrid> scr3 = grid.emptyDuplicate();
            cl_mem scr3Real = scr3->getRealGPUBuffer();
            cl_mem scr3Reci = scr3->getReciGPUBuffer();
            
            /*
             * multiply with the WGC kernel expansion
             */
            scr1->markReciGPUClean();
            scr1->markRealGPUDirty();
            scr2->markReciGPUClean();
            scr2->markRealGPUDirty();
            scr3->markReciGPUClean();
            scr3->markRealGPUDirty();
            
            err  = clSetKernelArg(wgc2ndE2Kernel, 0, sizeof(cl_mem), &scr1Reci);
            err |= clSetKernelArg(wgc2ndE2Kernel, 1, sizeof(cl_mem), &scr2Reci);
            err |= clSetKernelArg(wgc2ndE2Kernel, 2, sizeof(cl_mem), &scr3Reci);
            err |= clSetKernelArg(wgc2ndE2Kernel, 3, sizeof(cl_mem), &_keGPUMem0th);
            err |= clSetKernelArg(wgc2ndE2Kernel, 4, sizeof(cl_mem), &_keGPUMem1st);
            err |= clSetKernelArg(wgc2ndE2Kernel, 5, sizeof(cl_mem), &_keGPUMem2nd);
            err |= clSetKernelArg(wgc2ndE2Kernel, 6, sizeof(cl_mem), &_keGPUMem3rd);
            err |= clSetKernelArg(wgc2ndE2Kernel, 7, sizeof(cl_mem), &densityAReci);
            if(err != CL_SUCCESS){
                cerr << "ERROR in setting a WGC 2nd E kernel argument: " << err << endl;
                throw runtime_error("Failed to set a WGC 2nd E kernel argument.");
            }
        
            // enqueue the kernel over the entire "1D'd" grid
            err = clEnqueueNDRangeKernel(_queue, wgc2ndE2Kernel, 1, NULL, &totReciSize, NULL, 0, NULL, NULL);
            if (err != CL_SUCCESS){
                throw runtime_error("1D-enqueue of WGC 2nd E kernel failed.");
            }
                    
        
            // we can delete densityA and instead allocate densityB, to avoid any background FFT
            workGridA.reset();
            unique_ptr<CartesianOCLOOPGrid> scr4 = grid.emptyDuplicate();
            cl_mem scr4Real = scr4->getRealGPUBuffer();
            
            /*
             * get the scr1 and scr2 into real space, we only need them to be readable
             */
            scr1->enqueueBackwardTransform();
            scr2->enqueueBackwardTransform();
            scr3->enqueueBackwardTransform();
            
            /*
             * the last kernel for 1st order WGC 
             */
            scr4->markRealGPUClean();
            scr4->markReciGPUDirty();
            
            err  = clSetKernelArg(wgc2ndE3Kernel, 0, sizeof(cl_mem), &scr1Real);
            err |= clSetKernelArg(wgc2ndE3Kernel, 1, sizeof(cl_mem), &scr2Real);
            err |= clSetKernelArg(wgc2ndE3Kernel, 2, sizeof(cl_mem), &scr3Real);
            err |= clSetKernelArg(wgc2ndE3Kernel, 3, sizeof(cl_mem), &scr4Real);
            err |= clSetKernelArg(wgc2ndE3Kernel, 4, sizeof(cl_mem), &densReal);
            err |= clSetKernelArg(wgc2ndE3Kernel, 5, sizeof(cl_mem), &thetaReal);
            err |= clSetKernelArg(wgc2ndE3Kernel, 6, sizeof(double), &_beta);
            if(err != CL_SUCCESS){
                cerr << "ERROR in setting a WGC 3rd E kernel argument: " << err << endl;
                throw runtime_error("Failed to set a WGC 3rd E kernel argument.");
            }
        
            // enqueue the kernel over the entire "1D'd" grid
            err = clEnqueueNDRangeKernel(_queue, wgc2ndE3Kernel, 1, NULL, &enqSize, NULL, 0, NULL, NULL);
            if (err != CL_SUCCESS){
                throw runtime_error("1D-enqueue of WGC 3rd E kernel failed.");
            }
        
            const double energy = _CTF*scr4->integrate();
        
            scr1.reset();
            scr2.reset();
            scr3.reset();
            scr4.reset();
            density.reset();
            theta.reset();
            
            return energy;
        }
    
        throw runtime_error("XXX implement me");
    }

    double calcPotential(const CartesianOCLOOPGrid& grid, CartesianOCLOOPGrid& potential) const {
    
        if(!_sndOrder && !_vacCutoff){
            // first order Taylor expansion, no vacuum cutoff employed
            
            unique_ptr<CartesianOCLOOPGrid> dens = grid.duplicate();
            dens->transferRealToGPU();
        
            unique_ptr<CartesianOCLOOPGrid> workGridA = grid.emptyDuplicate();
            unique_ptr<CartesianOCLOOPGrid> workGridB = grid.emptyDuplicate();
            unique_ptr<CartesianOCLOOPGrid> scr1 = grid.emptyDuplicate();
            unique_ptr<CartesianOCLOOPGrid> scr2 = grid.emptyDuplicate();
            
            const size_t totRealSize = workGridA->getTotalGridPoints();
            const size_t enqSize = totRealSize/grid.getVectortypeAlignment();
            const size_t totReciSize = _kernel0th->n_rows * _kernel0th->n_slices * _kernel0th->n_cols;
            
            const cl_mem densReal = dens->getRealGPUBuffer();
            cl_mem densityAReal = workGridA->getRealGPUBuffer();
            cl_mem densityBReal = workGridB->getRealGPUBuffer();
            cl_mem scr1Real = scr1->getRealGPUBuffer();
            cl_mem scr2Real = scr2->getRealGPUBuffer();
            
            cl_mem densityAReci = workGridA->getReciGPUBuffer();
            cl_mem densityBReci = workGridB->getReciGPUBuffer();
            cl_mem scr1Reci = scr1->getReciGPUBuffer();
            cl_mem scr2Reci = scr2->getReciGPUBuffer();
            
            cl_int err;
            /*
             * First WGC 1st order energy kernel
             */
            workGridA->markRealGPUClean();
            workGridA->markReciGPUDirty();
            scr1->markRealGPUClean();
            scr1->markReciGPUDirty();
            
            err  = clSetKernelArg(wgc1stE1Kernel, 0, sizeof(cl_mem), &densReal);
            err |= clSetKernelArg(wgc1stE1Kernel, 1, sizeof(cl_mem), &densityAReal);
            err |= clSetKernelArg(wgc1stE1Kernel, 2, sizeof(cl_mem), &scr1Real);
            err |= clSetKernelArg(wgc1stE1Kernel, 3, sizeof(double), &_alpha);
            err |= clSetKernelArg(wgc1stE1Kernel, 4, sizeof(double), &_rhoS);
            if(err != CL_SUCCESS){
                cerr << "ERROR in setting a WGC 1st E kernel argument: " << err << endl;
                throw runtime_error("Failed to set a WGC 1st E kernel argument.");
            }
        
            // enqueue the kernel over the entire "1D'd" grid
            err = clEnqueueNDRangeKernel(_queue, wgc1stE1Kernel, 1, NULL, &enqSize, NULL, 0, NULL, NULL);
            if (err != CL_SUCCESS){
                throw runtime_error("1D-enqueue of WGC 1st E kernel failed.");
            }
        
            /*
             * move densA and scr1 into g space, scr2 is whatever
             */
            workGridA->enqueueForwardTransform();
            scr1->enqueueForwardTransform();
            
            /*
             * multiply with the WGC kernel expansion
             */
            scr1->markReciGPUClean();
            scr1->markRealGPUDirty();
            scr2->markReciGPUClean();
            scr2->markRealGPUDirty();
            
            err  = clSetKernelArg(wgc1stE2Kernel, 0, sizeof(cl_mem), &scr1Reci);
            err |= clSetKernelArg(wgc1stE2Kernel, 1, sizeof(cl_mem), &scr2Reci);
            err |= clSetKernelArg(wgc1stE2Kernel, 2, sizeof(cl_mem), &_keGPUMem0th);
            err |= clSetKernelArg(wgc1stE2Kernel, 3, sizeof(cl_mem), &densityAReci);
            err |= clSetKernelArg(wgc1stE2Kernel, 4, sizeof(cl_mem), &_keGPUMem1st);
            if(err != CL_SUCCESS){
                cerr << "ERROR in setting a WGC 2nd E kernel argument: " << err << endl;
                throw runtime_error("Failed to set a WGC 2nd E kernel argument.");
            }
        
            // enqueue the kernel over the entire "1D'd" grid
            err = clEnqueueNDRangeKernel(_queue, wgc1stE2Kernel, 1, NULL, &totReciSize, NULL, 0, NULL, NULL);
            if (err != CL_SUCCESS){
                throw runtime_error("1D-enqueue of WGC 2nd E kernel failed.");
            }
        
            /*
             * get the scr1 and scr2 into real space, we only need one of them to be writable
             */
            scr1->enqueueBackwardTransform();
            scr2->enqueueBackwardTransform();
                
            // we can delete densityA and instead allocate scr3, to avoid any background FFT
            workGridA.reset();
            unique_ptr<CartesianOCLOOPGrid> scr3 = grid.emptyDuplicate();
            cl_mem scr3Real = scr3->getRealGPUBuffer();
            
            /*
             * first potential-specific kernel
             */
            cl_mem potReal = potential.getRealGPUBuffer();
            
            workGridB->markRealGPUClean();
            workGridB->markReciGPUDirty();
            scr3->markRealGPUClean();
            scr3->markReciGPUDirty();
            scr1->markRealGPUClean();
            scr1->markReciGPUDirty();
            potential.markRealGPUClean();
            potential.markReciGPUDirty();
            
            err  = clSetKernelArg(wgc1stP1Kernel, 0, sizeof(cl_mem), &densReal);
            err |= clSetKernelArg(wgc1stP1Kernel, 1, sizeof(cl_mem), &scr1Real);
            err |= clSetKernelArg(wgc1stP1Kernel, 2, sizeof(cl_mem), &scr2Real);
            err |= clSetKernelArg(wgc1stP1Kernel, 3, sizeof(cl_mem), &scr3Real);
            err |= clSetKernelArg(wgc1stP1Kernel, 4, sizeof(cl_mem), &densityBReal);
            err |= clSetKernelArg(wgc1stP1Kernel, 5, sizeof(cl_mem), &potReal);
            err |= clSetKernelArg(wgc1stP1Kernel, 6, sizeof(double), &_beta);
            err |= clSetKernelArg(wgc1stP1Kernel, 7, sizeof(double), &_rhoS);
            if(err != CL_SUCCESS){
                cerr << "ERROR in setting a WGC 1st P kernel argument: " << err << endl;
                throw runtime_error("Failed to set a WGC 1st P kernel argument.");
            }
        
            // enqueue the kernel over the entire "1D'd" grid
            err = clEnqueueNDRangeKernel(_queue, wgc1stP1Kernel, 1, NULL, &enqSize, NULL, 0, NULL, NULL);
            if (err != CL_SUCCESS){
                throw runtime_error("1D-enqueue of WGC 1st P kernel failed.");
            }
                        
            const double energy = _CTF*scr3->integrate();
            
            /*
             * calculate the potential w.r.t. density**alpha
             */
            workGridB->enqueueForwardTransform();
            scr1->enqueueForwardTransform();
            scr2->resetToReciprocal(); // to avoid background FFT
            
            scr1->markReciGPUClean();
            scr1->markRealGPUDirty();
            scr2->markReciGPUClean();
            scr2->markRealGPUDirty();
            
            err  = clSetKernelArg(wgc1stE2Kernel, 0, sizeof(cl_mem), &scr1Reci);
            err |= clSetKernelArg(wgc1stE2Kernel, 1, sizeof(cl_mem), &scr2Reci);
            err |= clSetKernelArg(wgc1stE2Kernel, 2, sizeof(cl_mem), &_keGPUMem0th);
            err |= clSetKernelArg(wgc1stE2Kernel, 3, sizeof(cl_mem), &densityBReci);
            err |= clSetKernelArg(wgc1stE2Kernel, 4, sizeof(cl_mem), &_keGPUMem1st);
            if(err != CL_SUCCESS){
                cerr << "ERROR in setting a WGC 2nd E kernel argument: " << err << endl;
                throw runtime_error("Failed to set a WGC 2nd E kernel argument.");
            }
        
            // enqueue the kernel over the entire "1D'd" grid
            err = clEnqueueNDRangeKernel(_queue, wgc1stE2Kernel, 1, NULL, &totReciSize, NULL, 0, NULL, NULL);
            if (err != CL_SUCCESS){
                throw runtime_error("1D-enqueue of WGC 3rd E kernel failed.");
            }
            
            /*
             * last part of the potential
             */
            scr1->enqueueBackwardTransform();
            scr2->enqueueBackwardTransform();
            
            potential.markRealGPUClean();
            potential.markReciGPUDirty();
            
            err  = clSetKernelArg(wgc1stP2Kernel, 0, sizeof(cl_mem), &densReal);
            err |= clSetKernelArg(wgc1stP2Kernel, 1, sizeof(cl_mem), &scr1Real);
            err |= clSetKernelArg(wgc1stP2Kernel, 2, sizeof(cl_mem), &scr2Real);
            err |= clSetKernelArg(wgc1stP2Kernel, 3, sizeof(cl_mem), &potReal);
            err |= clSetKernelArg(wgc1stP2Kernel, 4, sizeof(double), &_alpha);
            err |= clSetKernelArg(wgc1stP2Kernel, 5, sizeof(double), &_CTF);
            if(err != CL_SUCCESS){
                cerr << "ERROR in setting a WGC 2nd P kernel argument: " << err << endl;
                throw runtime_error("Failed to set a WGC 2nd P kernel argument.");
            }
        
            // enqueue the kernel over the entire "1D'd" grid
            err = clEnqueueNDRangeKernel(_queue, wgc1stP2Kernel, 1, NULL, &enqSize, NULL, 0, NULL, NULL);
            if (err != CL_SUCCESS){
                throw runtime_error("1D-enqueue of WGC 2nd P kernel failed.");
            }
            
            scr1.reset();
            scr2.reset();
            scr3.reset();
            workGridB.reset();
            dens.reset();
                    
            return energy;
        } else if(_sndOrder && !_vacCutoff){
        
            // second order Tayler expansion, no vacuum cutoff
        
            unique_ptr<CartesianOCLOOPGrid> density = grid.duplicate();
            unique_ptr<CartesianOCLOOPGrid> workGridA = grid.duplicate();
            density->transferRealToGPU();
            workGridA->transferRealToGPU();
            
            unique_ptr<CartesianOCLOOPGrid> scr1 = grid.emptyDuplicate();
            unique_ptr<CartesianOCLOOPGrid> scr2 = grid.emptyDuplicate();
            unique_ptr<CartesianOCLOOPGrid> theta = grid.emptyDuplicate();
        
            const size_t totRealSize = workGridA->getTotalGridPoints();
            const size_t enqSize = totRealSize/grid.getVectortypeAlignment();
            const size_t totReciSize = _kernel0th->n_rows * _kernel0th->n_slices * _kernel0th->n_cols;
            
            const cl_mem densReal = density->getRealGPUBuffer();
            cl_mem densityAReal = workGridA->getRealGPUBuffer();
            cl_mem scr1Real = scr1->getRealGPUBuffer();
            cl_mem scr2Real = scr2->getRealGPUBuffer();
            cl_mem thetaReal = theta->getRealGPUBuffer();
            cl_mem potReal = potential.getRealGPUBuffer();
            
            cl_mem densityAReci = workGridA->getReciGPUBuffer();
            cl_mem scr1Reci = scr1->getReciGPUBuffer();
            cl_mem scr2Reci = scr2->getReciGPUBuffer();
            
            cl_int err;
            
            /*
             * assemble the first parts of the WGC expression
             */
            workGridA->markRealGPUClean();
            workGridA->markReciGPUDirty();
            scr1->markRealGPUClean();
            scr1->markReciGPUDirty();
            scr2->markRealGPUClean();
            scr2->markReciGPUDirty();
            theta->markRealGPUClean();
            theta->markReciGPUDirty();
            
            err  = clSetKernelArg(wgc2ndE1Kernel, 0, sizeof(cl_mem), &densityAReal);
            err |= clSetKernelArg(wgc2ndE1Kernel, 1, sizeof(cl_mem), &scr1Real);
            err |= clSetKernelArg(wgc2ndE1Kernel, 2, sizeof(cl_mem), &scr2Real);
            err |= clSetKernelArg(wgc2ndE1Kernel, 3, sizeof(cl_mem), &thetaReal);
            err |= clSetKernelArg(wgc2ndE1Kernel, 4, sizeof(double), &_alpha);
            err |= clSetKernelArg(wgc2ndE1Kernel, 5, sizeof(double), &_rhoS);
            if(err != CL_SUCCESS){
                cerr << "ERROR in setting a WGC 1st E kernel argument: " << err << endl;
                throw runtime_error("Failed to set a WGC 1st E kernel argument.");
            }
        
            // enqueue the kernel over the entire "1D'd" grid
            err = clEnqueueNDRangeKernel(_queue, wgc2ndE1Kernel, 1, NULL, &enqSize, NULL, 0, NULL, NULL);
            if (err != CL_SUCCESS){
                throw runtime_error("1D-enqueue of WGC 2nd E kernel failed.");
            }

            /*
             * move densA, scr1, and scr2 into g space
             */
            workGridA->enqueueForwardTransform();
            scr1->enqueueForwardTransform();
            scr2->enqueueForwardTransform();
                
            unique_ptr<CartesianOCLOOPGrid> scr3 = grid.emptyDuplicate();
            cl_mem scr3Real = scr3->getRealGPUBuffer();
            cl_mem scr3Reci = scr3->getReciGPUBuffer();
            
            /*
             * multiply with the WGC kernel expansion
             */
            scr1->markReciGPUClean();
            scr1->markRealGPUDirty();
            scr2->markReciGPUClean();
            scr2->markRealGPUDirty();
            scr3->markReciGPUClean();
            scr3->markRealGPUDirty();
            
            err  = clSetKernelArg(wgc2ndE2Kernel, 0, sizeof(cl_mem), &scr1Reci);
            err |= clSetKernelArg(wgc2ndE2Kernel, 1, sizeof(cl_mem), &scr2Reci);
            err |= clSetKernelArg(wgc2ndE2Kernel, 2, sizeof(cl_mem), &scr3Reci);
            err |= clSetKernelArg(wgc2ndE2Kernel, 3, sizeof(cl_mem), &_keGPUMem0th);
            err |= clSetKernelArg(wgc2ndE2Kernel, 4, sizeof(cl_mem), &_keGPUMem1st);
            err |= clSetKernelArg(wgc2ndE2Kernel, 5, sizeof(cl_mem), &_keGPUMem2nd);
            err |= clSetKernelArg(wgc2ndE2Kernel, 6, sizeof(cl_mem), &_keGPUMem3rd);
            err |= clSetKernelArg(wgc2ndE2Kernel, 7, sizeof(cl_mem), &densityAReci);
            if(err != CL_SUCCESS){
                cerr << "ERROR in setting a WGC 2nd E kernel argument: " << err << endl;
                throw runtime_error("Failed to set a WGC 2nd E kernel argument.");
            }
        
            // enqueue the kernel over the entire "1D'd" grid
            err = clEnqueueNDRangeKernel(_queue, wgc2ndE2Kernel, 1, NULL, &totReciSize, NULL, 0, NULL, NULL);
            if (err != CL_SUCCESS){
                throw runtime_error("1D-enqueue of WGC 2nd E kernel failed.");
            }
                    
        
            // we can delete densityA and instead allocate densityB, to avoid any background FFT
            workGridA.reset();
            unique_ptr<CartesianOCLOOPGrid> workGridB = grid.emptyDuplicate();
            cl_mem densityBReal = workGridB->getRealGPUBuffer();
            cl_mem densityBReci = workGridB->getReciGPUBuffer();
            
            /*
             * get the scr1 and scr2 into real space, we only need them to be readable
             */
            scr1->enqueueBackwardTransform();
            scr2->enqueueBackwardTransform();
            scr3->enqueueBackwardTransform();
            
            /*
             * setup grid for energy integration and first half of the potential
             */
            workGridB->markRealGPUClean();
            workGridB->markReciGPUDirty();
            potential.markRealGPUClean();
            potential.markReciGPUDirty();
            scr1->markRealGPUClean();
            scr1->markReciGPUDirty();
            scr2->markRealGPUClean();
            scr2->markReciGPUDirty();
            scr3->markRealGPUClean();
            scr3->markReciGPUDirty();
            
            err  = clSetKernelArg(wgc2ndP1Kernel, 0, sizeof(cl_mem), &densReal);
            err |= clSetKernelArg(wgc2ndP1Kernel, 1, sizeof(cl_mem), &densityBReal);
            err |= clSetKernelArg(wgc2ndP1Kernel, 2, sizeof(cl_mem), &thetaReal);
            err |= clSetKernelArg(wgc2ndP1Kernel, 3, sizeof(cl_mem), &scr1Real);
            err |= clSetKernelArg(wgc2ndP1Kernel, 4, sizeof(cl_mem), &scr2Real);
            err |= clSetKernelArg(wgc2ndP1Kernel, 5, sizeof(cl_mem), &scr3Real);
            err |= clSetKernelArg(wgc2ndP1Kernel, 6, sizeof(cl_mem), &potReal);
            err |= clSetKernelArg(wgc2ndP1Kernel, 7, sizeof(double), &_beta);
            if(err != CL_SUCCESS){
                cerr << "ERROR in setting a WGC 1st P kernel argument: " << err << endl;
                throw runtime_error("Failed to set a WGC 1st P kernel argument.");
            }
        
            // enqueue the kernel over the entire "1D'd" grid
            err = clEnqueueNDRangeKernel(_queue, wgc2ndP1Kernel, 1, NULL, &enqSize, NULL, 0, NULL, NULL);
            if (err != CL_SUCCESS){
                throw runtime_error("1D-enqueue of WGC 1st P kernel failed.");
            }
                    
            const double energy = _CTF*scr3->integrate();
        
            /*
             * get density**beta and the th*densB grid to the reciprocal space
             */
            workGridB->enqueueForwardTransform();
            scr1->enqueueForwardTransform(); // contains theta*densB
            scr2->enqueueForwardTransform(); // contains theta*theta*densB*0.5
                    
            /*
             * scr3 can be reused, we need it in the reciprocal space
             */
            scr3->resetToReciprocal(); // to avoid background FFT
            
            scr1->markReciGPUClean();
            scr1->markRealGPUDirty();
            scr2->markReciGPUClean();
            scr2->markRealGPUDirty();
            scr3->markReciGPUClean();
            scr3->markRealGPUDirty();
            
            err  = clSetKernelArg(wgc2ndP3Kernel, 0, sizeof(cl_mem), &scr1Reci);
            err |= clSetKernelArg(wgc2ndP3Kernel, 1, sizeof(cl_mem), &scr2Reci);
            err |= clSetKernelArg(wgc2ndP3Kernel, 2, sizeof(cl_mem), &scr3Reci);
            err |= clSetKernelArg(wgc2ndP3Kernel, 3, sizeof(cl_mem), &_keGPUMem0th);
            err |= clSetKernelArg(wgc2ndP3Kernel, 4, sizeof(cl_mem), &_keGPUMem1st);
            err |= clSetKernelArg(wgc2ndP3Kernel, 5, sizeof(cl_mem), &_keGPUMem2nd);
            err |= clSetKernelArg(wgc2ndP3Kernel, 6, sizeof(cl_mem), &_keGPUMem3rd);
            err |= clSetKernelArg(wgc2ndP3Kernel, 7, sizeof(cl_mem), &densityBReci);
            if(err != CL_SUCCESS){
                cerr << "ERROR in setting a WGC 3rd P kernel argument: " << err << endl;
                throw runtime_error("Failed to set a WGC 3rd P kernel argument.");
            }
                    
            // enqueue the kernel over the entire "1D'd" grid
            err = clEnqueueNDRangeKernel(_queue, wgc2ndP3Kernel, 1, NULL, &totReciSize, NULL, 0, NULL, NULL);
            if (err != CL_SUCCESS){
                throw runtime_error("1D-enqueue of WGC 3rd P kernel failed.");
            }
                    
            /*
             * get them over to real space again
             */
            scr3->enqueueBackwardTransform(); // 0th-order kernel * density**beta + 1st-order kernel * [theta*densB] + 2nd-order kernel * [theta*theta*densB*0.5]
            scr1->enqueueBackwardTransform(); // 1st-order kernel * density**beta + 3rd-order kernel * [theta*densB]
            scr2->enqueueBackwardTransform(); // 2nd-order kernel * density**beta
        
            /*
             * assemble the second half of the potential
             */
            err  = clSetKernelArg(wgc2ndP2Kernel, 0, sizeof(cl_mem), &densReal);
            err |= clSetKernelArg(wgc2ndP2Kernel, 1, sizeof(cl_mem), &scr1Real);
            err |= clSetKernelArg(wgc2ndP2Kernel, 2, sizeof(cl_mem), &scr2Real);
            err |= clSetKernelArg(wgc2ndP2Kernel, 3, sizeof(cl_mem), &scr3Real);
            err |= clSetKernelArg(wgc2ndP2Kernel, 4, sizeof(cl_mem), &thetaReal);
            err |= clSetKernelArg(wgc2ndP2Kernel, 5, sizeof(cl_mem), &potReal);
            err |= clSetKernelArg(wgc2ndP2Kernel, 6, sizeof(double), &_alpha);
            err |= clSetKernelArg(wgc2ndP2Kernel, 7, sizeof(double), &_CTF);
            if(err != CL_SUCCESS){
                cerr << "ERROR in setting a WGC 2nd P kernel argument: " << err << endl;
                throw runtime_error("Failed to set a WGC 2nd P kernel argument.");
            }
        
            // enqueue the kernel over the entire "1D'd" grid
            err = clEnqueueNDRangeKernel(_queue, wgc2ndP2Kernel, 1, NULL, &enqSize, NULL, 0, NULL, NULL);
            if (err != CL_SUCCESS){
                throw runtime_error("1D-enqueue of WGC 2nd P kernel failed.");
            }
            
            scr1.reset();
            scr2.reset();
            scr3.reset();
            workGridB.reset();
            theta.reset();
            density.reset();
            
            return energy;
        }
    
        throw runtime_error("XXX implement me");
    }

    unique_ptr<StressTensor> calcStress(const CartesianOCLOOPGrid& grid) const {
        throw runtime_error("XXX implement me");
    }

    
private:
    const double _CTF = 2.87123400018819;
    const double _alpha;
    const double _beta;
    const double _rhoS;
    const bool _sndOrder;
    const bool _vacCutoff;
    const cube * _kernel0th;
    const cube * _kernel1st;
    const cube * _kernel2nd;
    const cube * _kernel3rd;
    
    cl_context _ctx;
    cl_command_queue _queue;
    
    cl_mem _keGPUMem0th;
    cl_mem _keGPUMem1st;
    cl_mem _keGPUMem2nd;
    cl_mem _keGPUMem3rd;
    
    cl_program wgcOCLProg;
    
    cl_kernel wgc1stE1Kernel;
    size_t _localWGC1stE1Size;
    
    cl_kernel wgc1stE2Kernel;
    size_t _localWGC1stE2Size;
    
    cl_kernel wgc1stE3Kernel;
    size_t _localWGC1stE3Size;
    
    cl_kernel wgc1stP1Kernel;
    size_t _localWGC1stP1Size;
    
    cl_kernel wgc1stP2Kernel;
    size_t _localWGC1stP2Size;
    
    cl_kernel wgc2ndE1Kernel;
    size_t _localWGC2ndE1Size;
    
    cl_kernel wgc2ndE2Kernel;
    size_t _localWGC2ndE2Size;
    
    cl_kernel wgc2ndE3Kernel;
    size_t _localWGC2ndE3Size;
    
    cl_kernel wgc2ndP1Kernel;
    size_t _localWGC2ndP1Size;
    
    cl_kernel wgc2ndP2Kernel;
    size_t _localWGC2ndP2Size;
    
    cl_kernel wgc2ndP3Kernel;
    size_t _localWGC2ndP3Size;
    
    
    void setupKernels(CartesianOCLOOPGrid* example, const bool secondOrder){
        
        _ctx = example->getGPUContext();
        clRetainContext(_ctx);
        _queue = example->getGPUQueue();
        clRetainCommandQueue(_queue);
        cl_device_id* _devices = example->getGPUDevices();
        cl_uint noDevices = example->getNoGPUDevices();

	const string st1 =                                      "\n" \
"__kernel void wgc1stE_1(  __global const KEDFOCLV *dens,        \n" \
"    __global KEDFOCLV *densAl, global KEDFOCLV *data,           \n" \
"    double const alpha, const double rhoS){                     \n" \
"    const size_t idx = get_global_id(0);                        \n" \
"    const KEDFOCLV densA = KEDFPOWR(dens[idx], alpha);          \n" \
"    densAl[idx] = densA;                                        \n" \
"    data[idx] = (dens[idx] - 2*rhoS) * densA;                   \n" \
"}                                                               \n" \
"__kernel void wgc1stE_2(__global double *scr1,                  \n" \
"    __global double *scr2, __global const double* ke0th,        \n" \
"    __global const double *densAl, __global const double* ke1st){ \n" \
"    const size_t idx = get_global_id(0);                        \n" \
"    const double reA = densAl[2*idx];                           \n" \
"    const double imA = densAl[2*idx+1];                         \n" \
"    const double re1 = scr1[2*idx];                             \n" \
"    const double im1 = scr1[2*idx+1];                           \n" \
"    scr1[2*idx] = ke0th[idx]*reA + ke1st[idx]*re1;              \n" \
"    scr1[2*idx+1] = ke0th[idx]*imA + ke1st[idx]*im1;            \n" \
"    scr2[2*idx] = ke1st[idx]*reA;                               \n" \
"    scr2[2*idx+1] = ke1st[idx]*imA;                             \n" \
"}                                                               \n" \
"__kernel void wgc1stE_3(__global const KEDFOCLV *dens,          \n" \
"   __global const KEDFOCLV *scr1,__global KEDFOCLV *scr2,       \n" \
"   const double beta){                                          \n" \
"    const size_t idx = get_global_id(0);                        \n" \
"    const KEDFOCLV densB = KEDFPOWR(dens[idx],beta);            \n" \
"    const KEDFOCLV s1 = scr1[idx];                              \n" \
"    const KEDFOCLV s2 = scr2[idx];                              \n" \
"    scr2[idx] = densB*(s1+dens[idx]*s2);                        \n" \
"}                                                               \n" \
"__kernel void wgc1stP_1(__global const KEDFOCLV *density,       \n" \
"    __global KEDFOCLV *scr1, __global const KEDFOCLV *scr2,     \n" \
"    __global KEDFOCLV *scr3, __global KEDFOCLV *densityB,       \n" \
"    __global KEDFOCLV *poten, const double beta,                \n" \
"    const double rhoS){                                         \n" \
"    const size_t idx = get_global_id(0);                        \n" \
"    const KEDFOCLV dens = density[idx];                         \n" \
"    const KEDFOCLV densBM1 = KEDFPOWR(dens, beta-1);            \n" \
"    const KEDFOCLV densB   = densBM1*dens;                      \n" \
"    densityB[idx] = densB;                                      \n" \
"    const KEDFOCLV s1 = scr1[idx];                              \n" \
"    const KEDFOCLV s2 = scr2[idx];                              \n" \
"    const KEDFOCLV s3 = densB*(s1 + dens*s2);                   \n" \
"    scr3[idx] = s3;                                             \n" \
"    poten[idx] = beta*densBM1 * s1 + (beta+1) * densB*s2;       \n" \
"    scr1[idx] = (dens - 2*rhoS) * densB;                        \n" \
"}                                                               \n" \
"__kernel void wgc1stP_2(__global const KEDFOCLV *density,       \n" \
"    __global const KEDFOCLV *scr1, __global const KEDFOCLV *scr2, \n" \
"    __global KEDFOCLV *poten, const double alpha, const double ctf){ \n" \
"    const size_t idx = get_global_id(0);                        \n" \
"    const KEDFOCLV dens = density[idx];                         \n" \
"    const KEDFOCLV s1 = scr1[idx];                              \n" \
"    const KEDFOCLV s2 = scr2[idx];                              \n" \
"    const KEDFOCLV densAM1 = KEDFPOWR(dens,alpha-1);            \n" \
"    const KEDFOCLV densAlpha = densAM1*dens;                    \n" \
"    poten[idx] += alpha*densAM1 * s1 + (alpha+1)*densAlpha*s2;  \n" \
"    poten[idx] = poten[idx]*ctf;                                \n" \
"}                                                               \n" \
                                                                "\n";
        
        const string st2 =                                      "\n" \
"__kernel void wgc2ndE_1(  __global KEDFOCLV *densA,             \n" \
"    __global KEDFOCLV *scr1, global KEDFOCLV *scr2,             \n" \
"    __global KEDFOCLV *theta, const double alpha, const double rhoS){ \n" \
"    const size_t idx = get_global_id(0);                        \n" \
"    const KEDFOCLV dens = densA[idx];                           \n" \
"    const KEDFOCLV densAl = KEDFPOWR(dens,alpha);               \n" \
"    densA[idx] = densAl;                                        \n" \
"    const KEDFOCLV th = dens - rhoS;                            \n" \
"    theta[idx] = th;                                            \n" \
"    scr1[idx] = th*densAl;                                      \n" \
"    scr2[idx] = th*th * densAl * 0.5;                           \n" \
"}                                                               \n" \
"__kernel void wgc2ndE_2(__global double *scr1,                  \n" \
"    __global double *scr2, __global double *scr3,               \n" \
"    __global const double *ke0th, __global const double *ke1st, \n" \
"    __global const double *ke2nd, __global const double *ke3rd, \n" \
"    __global const double *recA){                               \n" \
"    const size_t idx = get_global_id(0);                        \n" \
"    const double reA = recA[2*idx];                             \n" \
"    const double imA = recA[2*idx+1];                           \n" \
"    const double re1 = scr1[2*idx];                             \n" \
"    const double im1 = scr1[2*idx+1];                           \n" \
"    const double re2 = scr2[2*idx];                             \n" \
"    const double im2 = scr2[2*idx+1];                           \n" \
"    scr1[2*idx] = ke0th[idx]*reA+ke1st[idx]*re1+ke2nd[idx]*re2; \n" \
"    scr1[2*idx+1] =ke0th[idx]*imA+ke1st[idx]*im1+ke2nd[idx]*im2;\n" \
"    scr2[2*idx] = ke1st[idx]*reA + ke3rd[idx]*re1;              \n" \
"    scr2[2*idx+1] = ke1st[idx]*imA + ke3rd[idx]*im1;            \n" \
"    scr3[2*idx] = ke2nd[idx]*reA;                               \n" \
"    scr3[2*idx+1] = ke2nd[idx]*imA;                             \n" \
"}                                                               \n" \
"__kernel void wgc2ndE_3(__global const KEDFOCLV *scr1,          \n" \
"    __global const KEDFOCLV *scr2, __global const KEDFOCLV *scr3, \n" \
"    __global KEDFOCLV *scr4, __global const KEDFOCLV *density,  \n" \
"    __global const KEDFOCLV *theta, const double beta){         \n" \
"    const size_t idx = get_global_id(0);                        \n" \
"    const KEDFOCLV densB = KEDFPOWR(density[idx],beta);         \n" \
"    const KEDFOCLV th = theta[idx];                             \n" \
"    scr4[idx] = densB*(scr1[idx] + th * scr2[idx]               \n" \
"            + th*th*0.5*scr3[idx]);                             \n" \
"}                                                               \n" \
"__kernel void wgc2ndP_1(__global const KEDFOCLV *density,       \n" \
"    __global KEDFOCLV *densityB, __global const KEDFOCLV *theta, \n" \
"    __global KEDFOCLV *scr1, __global KEDFOCLV *scr2,           \n" \
"    __global KEDFOCLV *scr3, __global KEDFOCLV *pot,            \n" \
"    const double beta){                                         \n" \
"    const size_t idx = get_global_id(0);                        \n" \
"    const KEDFOCLV dens = density[idx];                         \n" \
"    const KEDFOCLV th = theta[idx];                             \n" \
"    const KEDFOCLV s1 = scr1[idx];                              \n" \
"    const KEDFOCLV s2 = scr2[idx];                              \n" \
"    const KEDFOCLV s3 = scr3[idx];                              \n" \
"    const KEDFOCLV densBM1 = KEDFPOWR(dens,beta-1);             \n" \
"    const KEDFOCLV densB = densBM1*dens;                        \n" \
"    densityB[idx] = densB;                                      \n" \
"    scr3[idx] = densB*(s1 + th*s2 + th*th*0.5*s3);              \n" \
"    pot[idx] = densBM1 * (beta*s1 + (dens + beta*th) * s2       \n" \
"                    + th*(dens+beta*th*0.5)*s3);                \n" \
"    scr1[idx] = th*densB;                                       \n" \
"    scr2[idx] = th*th * densB * 0.5;                            \n" \
"}                                                               \n" \
"__kernel void wgc2ndP_2(__global const KEDFOCLV *density,       \n" \
"    __global const KEDFOCLV *scr1, __global const KEDFOCLV *scr2, \n" \
"    __global const KEDFOCLV *scr3, __global const KEDFOCLV *theta, \n" \
"    __global KEDFOCLV *poten, const double alpha, const double ctf){ \n" \
"    const size_t idx = get_global_id(0);                        \n" \
"    const KEDFOCLV dens = density[idx];                         \n" \
"    const KEDFOCLV s1 = scr1[idx];                              \n" \
"    const KEDFOCLV s2 = scr2[idx];                              \n" \
"    const KEDFOCLV s3 = scr3[idx];                              \n" \
"    const KEDFOCLV th = theta[idx];                             \n" \
"    const KEDFOCLV densAM1 = KEDFPOWR(dens,alpha-1);            \n" \
"    const KEDFOCLV densAlpha = densAM1*dens;                    \n" \
"    poten[idx] += densAM1 * (alpha*s3 + (dens+alpha*th)*s1      \n" \
"             + th*(dens+alpha*th*0.5)*s2);                      \n" \
"    poten[idx] = poten[idx]*ctf;                                \n" \
"}                                                               \n" \
"__kernel void wgc2ndP_3(__global double *scr1,                  \n" \
"    __global double *scr2, __global double *scr3,               \n" \
"    __global const double *ke0th, __global const double *ke1st, \n" \
"    __global const double *ke2nd, __global const double *ke3rd, \n" \
"    __global const double *recB){                               \n" \
"    const size_t idx = get_global_id(0);                        \n" \
"    const double reB = recB[2*idx];                             \n" \
"    const double imB = recB[2*idx+1];                           \n" \
"    const double re1 = scr1[2*idx];                             \n" \
"    const double im1 = scr1[2*idx+1];                           \n" \
"    const double re2 = scr2[2*idx];                             \n" \
"    const double im2 = scr2[2*idx+1];                           \n" \
"    scr3[2*idx] = ke0th[idx]*reB+ke1st[idx]*re1+ke2nd[idx]*re2; \n" \
"    scr3[2*idx+1] =ke0th[idx]*imB+ke1st[idx]*im1+ke2nd[idx]*im2;\n" \
"    scr1[2*idx] = ke1st[idx]*reB + ke3rd[idx]*re1;              \n" \
"    scr1[2*idx+1] = ke1st[idx]*imB + ke3rd[idx]*im1;            \n" \
"    scr2[2*idx] = ke2nd[idx]*reB;                               \n" \
"    scr2[2*idx+1] = ke2nd[idx]*imB;                             \n" \
"}                                                               \n" \
                                                                "\n";
        
        cl_int err;
        
        if(!secondOrder){
            
            const string macros = example->getMacroDefinitions();
    
            ostringstream os;
            os << macros;
            os << st1;

            const string s = os.str();
            const char* wgc1stOCLT = s.c_str();
            
            this->wgcOCLProg = clCreateProgramWithSource(_ctx, 1, (const char**) &wgc1stOCLT, NULL, &err);
            if(!wgcOCLProg || err != CL_SUCCESS){
                cerr << "ERROR to create WGC OCL program " << err << endl;
                throw runtime_error("Failed to create WGC 1st order OCL program.");
            }
            err = clBuildProgram(wgcOCLProg, noDevices, _devices, example->getCompilationOptions(), NULL, NULL);
            if(err != CL_SUCCESS){
                cerr << "ERROR in building WGC 1st order program " << err << endl;
            
                cl_build_status status;
                // check build error and build status first
                clGetProgramBuildInfo(wgcOCLProg, _devices[0], CL_PROGRAM_BUILD_STATUS, 
                    sizeof(cl_build_status), &status, NULL);
                
                if(!(err == CL_BUILD_PROGRAM_FAILURE && status == 0)){ // this is what I observe currently on NVIDIA and it seems to be caused by the KEDFOCL macro, makes no sense
 
                    // check build log
                    size_t logSize;
                    clGetProgramBuildInfo(wgcOCLProg, _devices[0], 
                            CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);
                    char* programLog = (char*) calloc (logSize+1, sizeof(char));
                    clGetProgramBuildInfo(wgcOCLProg, _devices[0],
                            CL_PROGRAM_BUILD_LOG, logSize+1, programLog, NULL);
                    printf("Build failed; error=%d, status=%d, programLog:nn%s \n", 
                            err, status, programLog);
                    free(programLog);
            
                    throw runtime_error("Could not build 1st order WGC program.");
                }
            }
        
            this->wgc1stE1Kernel = clCreateKernel(wgcOCLProg, "wgc1stE_1", &err);
            if(err != CL_SUCCESS){
                cerr << "ERROR in creating 1st WGC 1st order E kernel " << err << endl;
                throw runtime_error("Could not create 1st WGC 1st order E kernel.");
            }
        
            err = clGetKernelWorkGroupInfo(wgc1stE1Kernel, _devices[0], CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &_localWGC1stE1Size, NULL);
            if (err != CL_SUCCESS){
                cerr << "ERROR to inquire local work group size for 1st WGC 1st order E kernel " << err << endl;
                throw runtime_error("Not possible to inquire local work group size of 1st WGC 1st order E kernel.");
            }
            
            this->wgc1stE2Kernel = clCreateKernel(wgcOCLProg, "wgc1stE_2", &err);
            if(err != CL_SUCCESS){
                cerr << "ERROR in creating 2nd WGC 1st order E kernel " << err << endl;
                throw runtime_error("Could not create 2nd WGC 1st order E kernel.");
            }
        
            err = clGetKernelWorkGroupInfo(wgc1stE2Kernel, _devices[0], CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &_localWGC1stE2Size, NULL);
            if (err != CL_SUCCESS){
                cerr << "ERROR to inquire local work group size for 2nd WGC 1st order E kernel " << err << endl;
                throw runtime_error("Not possible to inquire local work group size of 2nd WGC 1st order E kernel.");
            }
            
            this->wgc1stE3Kernel = clCreateKernel(wgcOCLProg, "wgc1stE_3", &err);
            if(err != CL_SUCCESS){
                cerr << "ERROR in creating 3rd WGC 1st order E kernel " << err << endl;
                throw runtime_error("Could not create 3rd WGC 1st order E kernel.");
            }
        
            err = clGetKernelWorkGroupInfo(wgc1stE3Kernel, _devices[0], CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &_localWGC1stE3Size, NULL);
            if (err != CL_SUCCESS){
                cerr << "ERROR to inquire local work group size for 3rd WGC 1st order E kernel " << err << endl;
                throw runtime_error("Not possible to inquire local work group size of 3rd WGC 1st order E kernel.");
            }
            
            this->wgc1stP1Kernel = clCreateKernel(wgcOCLProg, "wgc1stP_1", &err);
            if(err != CL_SUCCESS){
                cerr << "ERROR in creating 1st WGC 1st order P kernel " << err << endl;
                throw runtime_error("Could not create 1st WGC 1st order P kernel.");
            }
        
            err = clGetKernelWorkGroupInfo(wgc1stP1Kernel, _devices[0], CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &_localWGC1stP1Size, NULL);
            if (err != CL_SUCCESS){
                cerr << "ERROR to inquire local work group size for 1st WGC 1st order P kernel " << err << endl;
                throw runtime_error("Not possible to inquire local work group size of 1st WGC 1st order P kernel.");
            }
            
            this->wgc1stP2Kernel = clCreateKernel(wgcOCLProg, "wgc1stP_2", &err);
            if(err != CL_SUCCESS){
                cerr << "ERROR in creating 2nd WGC 1st order P kernel " << err << endl;
                throw runtime_error("Could not create 2nd WGC 1st order P kernel.");
            }
        
            err = clGetKernelWorkGroupInfo(wgc1stP2Kernel, _devices[0], CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &_localWGC1stP2Size, NULL);
            if (err != CL_SUCCESS){
                cerr << "ERROR to inquire local work group size for 2nd WGC 1st order P kernel " << err << endl;
                throw runtime_error("Not possible to inquire local work group size of 2nd WGC 1st order P kernel.");
            }
        } else {
            
            const string macros = example->getMacroDefinitions();
    
            ostringstream os;
            os << macros;
            os << st2;

            const string s = os.str();
            const char* wgc2ndOCLT = s.c_str();
            
            this->wgcOCLProg = clCreateProgramWithSource(_ctx, 1, (const char**) &wgc2ndOCLT, NULL, &err);
            if(!wgcOCLProg || err != CL_SUCCESS){
                cerr << "ERROR to create WGC OCL program " << err << endl;
                throw runtime_error("Failed to create WGC 2nd order OCL program.");
            }
            err = clBuildProgram(wgcOCLProg, noDevices, _devices, example->getCompilationOptions(), NULL, NULL);
            if(err != CL_SUCCESS){
                cerr << "ERROR in building WGC 2nd order program " << err << endl;
            
                cl_build_status status;
                // check build error and build status first
                clGetProgramBuildInfo(wgcOCLProg, _devices[0], CL_PROGRAM_BUILD_STATUS, 
                    sizeof(cl_build_status), &status, NULL);
                
                if(!(err == CL_BUILD_PROGRAM_FAILURE && status == 0)){ // this is what I observe currently on NVIDIA and it seems to be caused by the KEDFOCL macro, makes no sense
 
                    // check build log
                    size_t logSize;
                    clGetProgramBuildInfo(wgcOCLProg, _devices[0], 
                            CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);
                    char* programLog = (char*) calloc (logSize+1, sizeof(char));
                    clGetProgramBuildInfo(wgcOCLProg, _devices[0],
                            CL_PROGRAM_BUILD_LOG, logSize+1, programLog, NULL);
                    printf("Build failed; error=%d, status=%d, programLog:nn%s \n", 
                            err, status, programLog);
                    free(programLog);
                    
                    throw runtime_error("Could not build 2nd order WGC program.");
                }
            }
        
            this->wgc2ndE1Kernel = clCreateKernel(wgcOCLProg, "wgc2ndE_1", &err);
            if(err != CL_SUCCESS){
                cerr << "ERROR in creating 1st WGC 2nd order E kernel " << err << endl;
                throw runtime_error("Could not create 1st WGC 2nd order E kernel.");
            }
        
            err = clGetKernelWorkGroupInfo(wgc2ndE1Kernel, _devices[0], CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &_localWGC2ndE1Size, NULL);
            if (err != CL_SUCCESS){
                cerr << "ERROR to inquire local work group size for 1st WGC 2nd order E kernel " << err << endl;
                throw runtime_error("Not possible to inquire local work group size of 1st WGC 2nd order E kernel.");
            }
            
            this->wgc2ndE2Kernel = clCreateKernel(wgcOCLProg, "wgc2ndE_2", &err);
            if(err != CL_SUCCESS){
                cerr << "ERROR in creating 2nd WGC 2nd order E kernel " << err << endl;
                throw runtime_error("Could not create 2nd WGC 2nd order E kernel.");
            }
        
            err = clGetKernelWorkGroupInfo(wgc2ndE2Kernel, _devices[0], CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &_localWGC2ndE2Size, NULL);
            if (err != CL_SUCCESS){
                cerr << "ERROR to inquire local work group size for 2nd WGC 2nd order E kernel " << err << endl;
                throw runtime_error("Not possible to inquire local work group size of 2nd WGC 2nd order E kernel.");
            }
            
            this->wgc2ndE3Kernel = clCreateKernel(wgcOCLProg, "wgc2ndE_3", &err);
            if(err != CL_SUCCESS){
                cerr << "ERROR in creating 3rd WGC 2nd order E kernel " << err << endl;
                throw runtime_error("Could not create 3rd WGC 2nd order E kernel.");
            }
        
            err = clGetKernelWorkGroupInfo(wgc2ndE3Kernel, _devices[0], CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &_localWGC2ndE3Size, NULL);
            if (err != CL_SUCCESS){
                cerr << "ERROR to inquire local work group size for 3rd WGC 2nd order E kernel " << err << endl;
                throw runtime_error("Not possible to inquire local work group 2nd of 3rd WGC 1st order E kernel.");
            }
            
            this->wgc2ndP1Kernel = clCreateKernel(wgcOCLProg, "wgc2ndP_1", &err);
            if(err != CL_SUCCESS){
                cerr << "ERROR in creating 1st WGC 2nd order P kernel " << err << endl;
                throw runtime_error("Could not create 2nd WGC 1st order P kernel.");
            }
        
            err = clGetKernelWorkGroupInfo(wgc2ndP1Kernel, _devices[0], CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &_localWGC2ndP1Size, NULL);
            if (err != CL_SUCCESS){
                cerr << "ERROR to inquire local work group size for 1st WGC 2nd order P kernel " << err << endl;
                throw runtime_error("Not possible to inquire local work group size of 1st WGC 2nd order P kernel.");
            }
            
            this->wgc2ndP2Kernel = clCreateKernel(wgcOCLProg, "wgc2ndP_2", &err);
            if(err != CL_SUCCESS){
                cerr << "ERROR in creating 2nd WGC 2nd order P kernel " << err << endl;
                throw runtime_error("Could not create 2nd WGC 2nd order P kernel.");
            }
        
            err = clGetKernelWorkGroupInfo(wgc2ndP2Kernel, _devices[0], CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &_localWGC2ndP2Size, NULL);
            if (err != CL_SUCCESS){
                cerr << "ERROR to inquire local work group size for 2nd WGC 2nd order P kernel " << err << endl;
                throw runtime_error("Not possible to inquire local work group size of 2nd WGC 2nd order P kernel.");
            }
            
            this->wgc2ndP3Kernel = clCreateKernel(wgcOCLProg, "wgc2ndP_3", &err);
            if(err != CL_SUCCESS){
                cerr << "ERROR in creating 2nd WGC 3rd order P kernel " << err << endl;
                throw runtime_error("Could not create 2nd WGC 3rd order P kernel.");
            }
        
            err = clGetKernelWorkGroupInfo(wgc2ndP3Kernel, _devices[0], CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &_localWGC2ndP3Size, NULL);
            if (err != CL_SUCCESS){
                cerr << "ERROR to inquire local work group size for 2nd WGC 3rd order P kernel " << err << endl;
                throw runtime_error("Not possible to inquire local work group size of 3rd WGC 2nd order P kernel.");
            }
        }
    }
};

#endif /* TAYLOREDWANGGOVINDCARTEROCL_HPP */

