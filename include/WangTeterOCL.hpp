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

#ifndef WANGTETEROCL_HPP
#define WANGTETEROCL_HPP

#include <CL/cl.h>
#include <string>
#include "CartesianOCLOOPGrid.hpp"

template<>
class WangTeter<CartesianOCLOOPGrid>: public KEDF<CartesianOCLOOPGrid> {
    
public:
    WangTeter(CartesianOCLOOPGrid* example, const double alpha, const double beta, cube* keKernel)
        : _alpha(alpha), _beta(beta), _keKernel(keKernel){
        
        // setup internal kernels
        setupKernels(example);
        
        // push the keKernel to the GPU where it belongs
        const size_t totReciSize = _keKernel->n_rows * _keKernel->n_slices * _keKernel->n_cols;
        const size_t halfReciMem = totReciSize*sizeof(double);
        
        cl_int err;
        _keGPUMem = clCreateBuffer(_ctx, CL_MEM_READ_ONLY, halfReciMem, NULL, &err);
        if(err != CL_SUCCESS){
            cerr << "ERROR in allocating GPU memory of size " << halfReciMem << " with error " << err << endl;
            throw runtime_error("Failed to allocate GPU memory");
        }
        err = clEnqueueWriteBuffer(_queue, _keGPUMem, CL_FALSE, 0, halfReciMem, _keKernel->memptr(), 0, NULL, NULL );
        if(err != CL_SUCCESS){
            cerr << "ERROR to write kernel data to GPU " << err << endl;
            throw runtime_error("Failed to write kernel data to GPU.");
        }
        
        // create a on GPU scratch buffer
        const size_t totRealSize = example->getTotalGridPoints();
        const size_t totRealMem = sizeof(double)*totRealSize;
        _scrReal = clCreateBuffer(_ctx, CL_MEM_READ_WRITE, totRealMem, NULL, &err);
        if(err != CL_SUCCESS){
            cerr << "ERROR in allocating GPU memory of size " << totRealMem << " with error " << err << endl;
            throw runtime_error("Failed to allocate GPU memory");
        }
    }
    
    ~WangTeter() {
        clReleaseCommandQueue(_queue);
        clReleaseContext(_ctx);
        clReleaseMemObject(_scrReal);
        clReleaseMemObject(_keGPUMem);
        clReleaseProgram(wtOCLProg);
        clReleaseKernel(cmplRealMult);
        clReleaseKernel(wtE2ndKernel);
        clReleaseKernel(wtPowGKernel);
        clReleaseKernel(wtAssFHKernel);
        clReleaseKernel(wtAssSHKernel);
    }
    
    string getMethodDescription() const {
        return "Wang-Teter KEDF (OCL version)";
    }
    
    vector<string> getCitations() const {
    
        vector<string> citations(0);
        citations.push_back("L.-W. Wang and M. P. Teter, Phys. Rev. B 45, 13196 (1992).");
    
        return citations;
    }
    
    vector<string> getWorkingEquations() const {
    
        vector<string> equations(0);
        equations.push_back("");
    
        //   WTEnergy = cTF * SUM(rhoR_SI**alpha * FFT(FFT(rhoR_SI**beta) * keKernel(:,:,:,1)))*/
    
        return equations;
    }
    
    double calcEnergy(const CartesianOCLOOPGrid& grid) const {
        
        unique_ptr<CartesianOCLOOPGrid> workGridB = grid.duplicate();
        
        workGridB->transferRealToGPU();
        cl_mem realGPUmem = workGridB->getRealGPUBuffer();
        cl_mem reciGPUmem = workGridB->getReciGPUBuffer();
        
        // copy density
        const size_t totRealSize = workGridB->getTotalGridPoints();
        const size_t enqSize = totRealSize/grid.getVectortypeAlignment();
        const size_t totRealMem = sizeof(double)*totRealSize;
        
        cl_int err;
        err = clEnqueueCopyBuffer (_queue, realGPUmem, _scrReal, 0, 0, totRealMem, 0, NULL, NULL);
        if(err != CL_SUCCESS){
            cerr << "ERROR in copying density " << err << endl;
            throw runtime_error("Failed to copy density.");
        }
        
        workGridB->powGrid(_beta);
        
        const size_t totReciSize = _keKernel->n_rows * _keKernel->n_slices * _keKernel->n_cols;
        
        // do the convolusion w/ the keKernel
        workGridB->transferReciToGPU();
        
        err  = clSetKernelArg(cmplRealMult, 0, sizeof(cl_mem), &reciGPUmem);
        err |= clSetKernelArg(cmplRealMult, 1, sizeof(cl_mem), &_keGPUMem);
        if(err != CL_SUCCESS){
            cerr << "ERROR in setting a complex mult kernel argument: " << err << endl;
            throw runtime_error("Failed to set complex mult kernel argument.");
        }
        
        // enqueue the kernel over the entire "1D'd" grid
        err = clEnqueueNDRangeKernel(_queue, cmplRealMult, 1, NULL, &totReciSize, NULL, 0, NULL, NULL);
        if (err != CL_SUCCESS){
            cerr << "ERROR: Failed to enqueue complex mult kernel. (I) " << err << endl;
            throw runtime_error("1D-enqueue of complex mult kernel failed.");
        }
        
        workGridB->markRealDirty();
        workGridB->markReciDirty();
        
        workGridB->enqueueBackwardTransform();

        // multiply with the power of densA
        err  = clSetKernelArg(wtE2ndKernel, 0, sizeof(cl_mem), &realGPUmem);
        err |= clSetKernelArg(wtE2ndKernel, 1, sizeof(cl_mem), &_scrReal);
        err |= clSetKernelArg(wtE2ndKernel, 2, sizeof(double), &_alpha);
        if(err != CL_SUCCESS){
            cerr << "ERROR in setting a 2nd WT kernel argument: " << err << endl;
            throw runtime_error("Failed to set a 2nd WT kernel argument.");
        }
        
        // enqueue the kernel over the entire "1D'd" grid
        err = clEnqueueNDRangeKernel(_queue, wtE2ndKernel, 1, NULL, &enqSize, NULL, 0, NULL, NULL);
        if (err != CL_SUCCESS){
            cerr << "ERROR failed to enqueue 2nd WT kernel " << err << endl;
            throw runtime_error("1D-enqueue of 2nd WT kernel failed.");
        }
        
        workGridB->markRealDirty();
        workGridB->markReciDirty();

        const double eWT = _CTF * workGridB->integrate();
        
        workGridB.reset();

        return eWT;
    }
    
    double calcPotential(const CartesianOCLOOPGrid& grid, CartesianOCLOOPGrid& potential) const {
        
        unique_ptr<CartesianOCLOOPGrid> workGridA = grid.duplicate();
        
        workGridA->transferRealToGPU();
        cl_mem densAReal = workGridA->getRealGPUBuffer();
        cl_mem densAReci = workGridA->getReciGPUBuffer();
        cl_mem realPot = potential.getRealGPUBuffer();
        cl_mem reciPot = potential.getReciGPUBuffer();
        
        // copy density
        const size_t totRealSize = workGridA->getTotalGridPoints();
        const size_t enqSize = totRealSize/grid.getVectortypeAlignment();
        const size_t totRealMem = sizeof(double)*totRealSize;
        const size_t totReciSize = _keKernel->n_rows * _keKernel->n_slices * _keKernel->n_cols;
        
        cl_int err;
        err = clEnqueueCopyBuffer (_queue, densAReal, _scrReal, 0, 0, totRealMem, 0, NULL, NULL);
        if(err != CL_SUCCESS){
            cerr << "ERROR in copying density " << err << endl;
            throw runtime_error("Failed to copy density.");
        }
        
        // we need another temporary scratch space
        cl_mem densB = clCreateBuffer(_ctx, CL_MEM_READ_WRITE, totRealMem, NULL, &err);
        if(err != CL_SUCCESS){
            cerr << "ERROR in allocating GPU memory of size " << totRealMem << " with error " << err << endl;
            throw runtime_error("Failed to allocate GPU memory");
        }
        
        // DO ALL THE POWER OPERATIONS AS THE FIRST PART OF THE KERNEL
        // we do not need to transfer, but we need to tell it that the real data is just fine (it is garbage but we don't care))
        workGridA->markRealGPUClean();
        workGridA->markReciGPUDirty();
        potential.markRealGPUClean();
        potential.markReciGPUDirty();
        
        err  = clSetKernelArg(wtPowGKernel, 0, sizeof(cl_mem), &_scrReal);
        err |= clSetKernelArg(wtPowGKernel, 1, sizeof(cl_mem), &densB);
        err |= clSetKernelArg(wtPowGKernel, 2, sizeof(cl_mem), &realPot);
        err |= clSetKernelArg(wtPowGKernel, 3, sizeof(cl_mem), &densAReal);
        err |= clSetKernelArg(wtPowGKernel, 4, sizeof(double), &_alpha);
        const double betaM1 = _beta-1;
        err |= clSetKernelArg(wtPowGKernel, 5, sizeof(double), &betaM1);
        if(err != CL_SUCCESS){
            cerr << "ERROR in setting a WT pow G argument: " << err << endl;
            throw runtime_error("Failed to set a WT pow G kernel argument.");
        }
        
        // enqueue the kernel over the entire "1D'd" grid
        err = clEnqueueNDRangeKernel(_queue, wtPowGKernel, 1, NULL, &enqSize, NULL, 0, NULL, NULL);
        if (err != CL_SUCCESS){
            throw runtime_error("1D-enqueue of WT pow G kernel failed.");
        }
        
        potential.markRealDirty();
        potential.markReciDirty();
        workGridA->markRealDirty();
        workGridA->markReciDirty();
        
        // DO THE CONVOLUSION WITH THE KEKERNEL FOR THE POTENTIAL
        potential.transferReciToGPU();
        
        err  = clSetKernelArg(cmplRealMult, 0, sizeof(cl_mem), &reciPot);
        err |= clSetKernelArg(cmplRealMult, 1, sizeof(cl_mem), &_keGPUMem);
        if(err != CL_SUCCESS){
            cerr << "ERROR in setting a complex mult kernel argument: " << err << endl;
            throw runtime_error("Failed to set a complex mult kernel argument.");
        }
        
        // enqueue the kernel over the entire "1D'd" grid
        err = clEnqueueNDRangeKernel(_queue, cmplRealMult, 1, NULL, &totReciSize, NULL, 0, NULL, NULL);
        if (err != CL_SUCCESS){
            cerr << "ERROR: Failed to enqueue complex mult kernel. (II) " << err << endl;
            throw runtime_error("1D-enqueue of complex mult kernel failed.");
        }
        
        potential.markRealDirty();
        potential.markReciDirty();
    
        potential.enqueueBackwardTransform();
                
        potential.multiplyElementwise(workGridA.get());

        // GET THE ENERGY
        const double eWT = _CTF * potential.integrate();

        // TRANSFORM WHAT IS CURRENTLY IN POTENTIAL INTO THE FIRST PART OF THE ACTUAL POTENTIAL
        
        err  = clSetKernelArg(wtAssFHKernel, 0, sizeof(cl_mem), &_scrReal);
        err |= clSetKernelArg(wtAssFHKernel, 1, sizeof(cl_mem), &realPot);
        const double preAl = _CTF*_alpha;
        err |= clSetKernelArg(wtAssFHKernel, 2, sizeof(double), &preAl);
        if(err != CL_SUCCESS){
            cerr << "ERROR in setting a WT assemble FH  argument: " << err << endl;
            throw runtime_error("Failed to set a WT assemble FH kernel argument.");
        }
        
        // enqueue the kernel over the entire "1D'd" grid
        err = clEnqueueNDRangeKernel(_queue, wtAssFHKernel, 1, NULL, &enqSize, NULL, 0, NULL, NULL);
        if (err != CL_SUCCESS){
            throw runtime_error("1D-enqueue of WT assemble FH kernel failed.");
        }
                
        // DO THE CONVOLUSION WITH THE KEKERNEL
        workGridA->transferReciToGPU(); // this will FFT internally
        
        err  = clSetKernelArg(cmplRealMult, 0, sizeof(cl_mem), &densAReci);
        err |= clSetKernelArg(cmplRealMult, 1, sizeof(cl_mem), &_keGPUMem);
        if(err != CL_SUCCESS){
            cerr << "ERROR in setting a complex mult kernel argument: " << err << endl;
            throw runtime_error("Failed to set a complex mult kernel argument.");
        }
        
        // enqueue the kernel over the entire "1D'd" grid
        err = clEnqueueNDRangeKernel(_queue, cmplRealMult, 1, NULL, &totReciSize, NULL, 0, NULL, NULL);
        if (err != CL_SUCCESS){
            throw runtime_error("1D-enqueue of complex mult kernel failed.");
        }
        
        workGridA->markRealDirty();
        workGridA->markReciDirty();
    
        workGridA->enqueueBackwardTransform(); // we are back in real space after this
        
        // DO THE SECOND PART OF THE POTENTIAL ASSEMBLING
        err  = clSetKernelArg(wtAssSHKernel, 0, sizeof(cl_mem), &densAReal);
        err |= clSetKernelArg(wtAssSHKernel, 1, sizeof(cl_mem), &densB);
        err |= clSetKernelArg(wtAssSHKernel, 2, sizeof(cl_mem), &realPot);
        const double preBe = _CTF*_beta;
        err |= clSetKernelArg(wtAssSHKernel, 3, sizeof(double), &preBe);
        if(err != CL_SUCCESS){
            cerr << "ERROR in setting a WT assemble SH argument: " << err << endl;
            throw runtime_error("Failed to set a WT assemble SH kernel argument.");
        }
        
        // enqueue the kernel over the entire "1D'd" grid
        err = clEnqueueNDRangeKernel(_queue, wtAssSHKernel, 1, NULL, &enqSize, NULL, 0, NULL, NULL);
        if (err != CL_SUCCESS){
            throw runtime_error("1D-enqueue of WT assemble SH kernel failed.");
        }
        
        // discard scratch space
        clReleaseMemObject(densB);
        
        workGridA.reset();
        
        return eWT;
    }
    
    unique_ptr<StressTensor> calcStress(const CartesianOCLOOPGrid& grid) const {
        throw runtime_error("not yet implemented");
    }
    
private:

    const double _CTF = 2.87123400018819;
    const double _alpha;
    const double _beta;
    const cube * _keKernel;
    
    cl_context _ctx;
    cl_command_queue _queue;
    
    cl_mem _keGPUMem;
    cl_mem _scrReal;
    
    cl_kernel cmplRealMult;
    
    cl_program wtOCLProg;
    
    cl_kernel wtE2ndKernel;
    size_t _localWTE2ndSize;
    
    cl_kernel wtPowGKernel;
    size_t _localWTPowGSize;
    
    cl_kernel wtAssFHKernel;
    size_t _localWTAssFHSize;
    
    cl_kernel wtAssSHKernel;
    size_t _localWTAssSHSize;
    
    void setupKernels(CartesianOCLOOPGrid* example){
        
        _ctx = example->getGPUContext();
        clRetainContext(_ctx);
        _queue = example->getGPUQueue();
        clRetainCommandQueue(_queue);
        cl_device_id* _devices = example->getGPUDevices();
        cl_uint noDevices = example->getNoGPUDevices();
        cmplRealMult = example->getRealComplexMultKernel();

	const string st1 =                                      "\n" \
"__kernel void wt2ndE(  __global KEDFOCLV *data,                 \n" \
"    __global const KEDFOCLV *dens, const double alpha){         \n" \
"    const size_t idx = get_global_id(0);                        \n" \
"    data[idx] *= KEDFPOWR(dens[idx], alpha);                    \n" \
"}                                                               \n" \
"__kernel void wtPowG(  __global const KEDFOCLV *dens,           \n" \
"    __global KEDFOCLV *densB, __global KEDFOCLV* pot,           \n" \
"    __global KEDFOCLV *densA, const double alpha,               \n" \
"    const double betaM1){                                       \n" \
"    const size_t idx = get_global_id(0);                        \n" \
"    const KEDFOCLV rho = dens[idx];                             \n" \
"    const KEDFOCLV rhoBM1 = KEDFPOWR(rho, betaM1);              \n" \
"    densB[idx] = rhoBM1;                                        \n" \
"    pot[idx] = rhoBM1*rho;                                      \n" \
"    densA[idx] = KEDFPOWR(rho, alpha);                          \n" \
"}                                                               \n" \
"__kernel void wtAssFH(  __global const KEDFOCLV *dens,          \n" \
"    __global KEDFOCLV *data, const double preBe){               \n" \
"    const size_t idx = get_global_id(0);                        \n" \
"    data[idx] *= preBe/dens[idx];                               \n" \
"}                                                               \n" \
"__kernel void wtAssSH(  __global const KEDFOCLV *densA,         \n" \
"    __global const KEDFOCLV *densB, __global KEDFOCLV *pot,     \n" \
"    const double preBe){                                        \n" \
"    const size_t idx = get_global_id(0);                        \n" \
"    const KEDFOCLV potX = densA[idx]*preBe*densB[idx];          \n" \
"    pot[idx] += potX;                                           \n" \
"}                                                               \n" \
                                                                "\n";
        
        const string macros = example->getMacroDefinitions();
    
        ostringstream os;
        os << macros;
        os << st1;

        const string s = os.str();
        const char* wtOCLT = s.c_str();
        
        cl_int err;
        this->wtOCLProg = clCreateProgramWithSource(_ctx, 1, (const char**) &wtOCLT, NULL, &err);
        if(!wtOCLProg || err != CL_SUCCESS){
            cerr << "ERROR to create WT OCL program " << err << endl;
            throw runtime_error("Failed to create WT OCL program.");
        }
        err = clBuildProgram(wtOCLProg, noDevices, _devices, example->getCompilationOptions(), NULL, NULL);
        if(err != CL_SUCCESS){
            cerr << "ERROR in building 2nd WT program " << err << endl;
            
            cl_build_status status;
            // check build error and build status first
            clGetProgramBuildInfo(wtOCLProg, _devices[0], CL_PROGRAM_BUILD_STATUS, 
                sizeof(cl_build_status), &status, NULL);
 
            if(!(err == CL_BUILD_PROGRAM_FAILURE && status == 0)){ // this is what I observe currently on NVIDIA and it seems to be caused by the KEDFOCL macro, makes no sense
                // check build log
                size_t logSize;
                clGetProgramBuildInfo(wtOCLProg, _devices[0], 
                        CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);
                char* programLog = (char*) calloc (logSize+1, sizeof(char));
                clGetProgramBuildInfo(wtOCLProg, _devices[0],
                        CL_PROGRAM_BUILD_LOG, logSize+1, programLog, NULL);
                printf("Build failed; error=%d, status=%d, programLog:nn%s \n", 
                        err, status, programLog);
                free(programLog);
            
                throw runtime_error("Could not build 2nd WT program.");
            }
        }
        
        this->wtE2ndKernel = clCreateKernel(wtOCLProg, "wt2ndE", &err);
        if(err != CL_SUCCESS){
            cerr << "ERROR in creating 2nd WT kernel " << err << endl;
            throw runtime_error("Could not create 2nd WT kernel.");
        }
        
        err = clGetKernelWorkGroupInfo(wtE2ndKernel, _devices[0], CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &_localWTE2ndSize, NULL);
        if (err != CL_SUCCESS){
            cerr << "ERROR to inquire local work group size for 2nd WT kernel " << err << endl;
            throw runtime_error("Not possible to inquire local work group size of 2nd WT kernel.");
        }

        this->wtPowGKernel = clCreateKernel(wtOCLProg, "wtPowG", &err);
        if(err != CL_SUCCESS){
            cerr << "ERROR in creating WT pow G kernel " << err << endl;
            throw runtime_error("Could not create WT pow G kernel.");
        }
        
        err = clGetKernelWorkGroupInfo(wtPowGKernel, _devices[0], CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &_localWTPowGSize, NULL);
        if (err != CL_SUCCESS){
            cerr << "ERROR to inquire local work group size for WT pow G kernel " << err << endl;
            throw runtime_error("Not possible to inquire local work group size of WT pow G kernel.");
        }
        
        
        // get the assembling of the first half of the potential kernel
        this->wtAssFHKernel = clCreateKernel(wtOCLProg, "wtAssFH", &err);
        if(err != CL_SUCCESS){
            cerr << "ERROR in creating WT assemble FH kernel " << err << endl;
            throw runtime_error("Could not create WT assemble FH kernel.");
        }
        
        err = clGetKernelWorkGroupInfo(wtAssFHKernel, _devices[0], CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &_localWTAssFHSize, NULL);
        if (err != CL_SUCCESS){
            cerr << "ERROR to inquire local work group size for WT assemble FH kernel " << err << endl;
            throw runtime_error("Not possible to inquire local work group size of WT assemble FH kernel.");
        }
        
        // get the assembling of the second half of the potential kernel        
        this->wtAssSHKernel = clCreateKernel(wtOCLProg, "wtAssSH", &err);
        if(err != CL_SUCCESS){
            cerr << "ERROR in creating WT assemble FH kernel " << err << endl;
            throw runtime_error("Could not create WT assemble SH kernel.");
        }
        
        err = clGetKernelWorkGroupInfo(wtAssSHKernel, _devices[0], CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &_localWTAssSHSize, NULL);
        if (err != CL_SUCCESS){
            cerr << "ERROR to inquire local work group size for WT assemble FH kernel " << err << endl;
            throw runtime_error("Not possible to inquire local work group size of WT assemble SH kernel.");
        }        
    }
};

#endif /* WANGTETEROCL_HPP */

