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

#ifndef VONWEIZSAECKEROCL_HPP
#define VONWEIZSAECKEROCL_HPP

#include <CL/cl.h>
#include <string>
#include "CartesianOCLOOPGrid.hpp"

template<>
class VonWeizsaecker<CartesianOCLOOPGrid>: public KEDF<CartesianOCLOOPGrid> {
    
public:

    VonWeizsaecker(CartesianOCLOOPGrid* example){
        setupKernels(example);
    }
    
    ~VonWeizsaecker(){
        clReleaseCommandQueue(_queue);
        clReleaseContext(_ctx);
        clReleaseProgram(vWOCLProg);
        clReleaseKernel(vWKPotKernel);
        clReleaseKernel(vWKStressKernel);
    }
    
    string getMethodDescription() const {
        return "von Weizsaecker KEDF (OpenCL version)";
    }
    
    vector<string> getCitations() const {
        
        vector<string> citations(0);
        citations.push_back("C. F. v. Weizsäcker, Z. Phys. 96, 431 (1935).");
    
        return citations;
    }
    
    vector<string> getWorkingEquations() const {
        
        vector<string> equations(0);
        equations.push_back("\\dfrac{1}{2}\\mu\\Sum\\left[\\right]");
    
        // VWEnergy = -0.5_DP * SUM(sqrtRhoR_SI*FFT(FFT(sqrtRhoR_SI)*qTable**2))
    
        return equations;
    }
    
    double calcEnergy(const CartesianOCLOOPGrid& grid) const {
        
        unique_ptr<CartesianOCLOOPGrid> workGrid(grid.duplicate());
        workGrid->sqrtGrid();

        const unique_ptr<CartesianOCLOOPGrid> laplacian(workGrid->laplacian());

        laplacian->multiplyElementwise(workGrid.get());
        
        const double energy = laplacian->integrate();
        
        workGrid.reset();

        return -0.5*energy;
    }
    
    double calcPotential(const CartesianOCLOOPGrid& grid, CartesianOCLOOPGrid& potential) const {
        
        unique_ptr<CartesianOCLOOPGrid> workGrid(grid.duplicate());
        workGrid->sqrtGrid();
    
        const unique_ptr<CartesianOCLOOPGrid> laplacian(workGrid->laplacian());
    
        // transfer the real density grid and laplacian to the GPU
        workGrid->transferRealToGPU();
        laplacian->transferRealToGPU();
        
        potential.markRealDirty();
        potential.markReciDirty();
        potential.markRealGPUClean();
        potential.markReciGPUDirty();
        
        // get the buffers and hook them into the kernel
        cl_mem dens = workGrid->getRealGPUBuffer();
        cl_mem pot = potential.getRealGPUBuffer();
        cl_mem lapl = laplacian->getRealGPUBuffer();
        
        cl_int err;
        err  = clSetKernelArg(vWKPotKernel, 0, sizeof(cl_mem), &dens);
        err |= clSetKernelArg(vWKPotKernel, 1, sizeof(cl_mem), &pot);
        err |= clSetKernelArg(vWKPotKernel, 2, sizeof(cl_mem), &lapl);
        if(err != CL_SUCCESS){
            cerr << "ERROR in setting a vW kernel argument: " << err << endl;
            throw runtime_error("Failed to set a vW kernel argument.");
        }
        
        // enqueue the kernel over the entire "1D'd" grid
        const size_t realPoints = workGrid->getTotalGridPoints();
        err = clEnqueueNDRangeKernel(_queue, vWKPotKernel, 1, NULL, &realPoints, NULL, 0, NULL, NULL);
        if (err != CL_SUCCESS){
            cerr << "ERROR: 1D-enqueue of vW potential potential kernel failed. " << err << endl;
            throw runtime_error("1D-enqueue of vW potential potential kernel failed.");
        }
        
        laplacian->multiplyElementwise(workGrid.get());
        
        const double energy = laplacian->integrate();
        
        workGrid.reset();
    
        return -0.5*energy;
    }
    
    unique_ptr<StressTensor> calcStress(const CartesianOCLOOPGrid& grid) const {
        
        unique_ptr<StressTensor> stress = make_unique<StressTensor>();
        mat* const tensor = stress->getTensor();    

        unique_ptr<CartesianOCLOOPGrid> workGrid(grid.duplicate());
        workGrid->sqrtGrid();

        // we will need each directional divergence three times: store them it is worth the memory
        unique_ptr<CartesianOCLOOPGrid> directionalDivs[3];
        directionalDivs[0] = workGrid->directionalDivergenceX();
        directionalDivs[1] = workGrid->directionalDivergenceY();
        directionalDivs[2] = workGrid->directionalDivergenceZ();

        const unsigned long long totalPoints = workGrid->getTotalGridPoints();

        // we only need to transfer the workgrid once to the GPU
        workGrid->transferRealToGPU();
        cl_mem gr = workGrid->getRealGPUBuffer();
        
        for(size_t i = 0; i < 3; ++i){

            directionalDivs[i]->transferRealToGPU();
            cl_mem divI = directionalDivs[i]->getRealGPUBuffer();

            for(size_t j = i; j < 3; ++j){

                directionalDivs[j]->transferRealToGPU();
                cl_mem divJ = directionalDivs[j]->getRealGPUBuffer();
                
        
                cl_int err;
                err  = clSetKernelArg(vWKStressKernel, 0, sizeof(cl_mem), &gr);
                err |= clSetKernelArg(vWKStressKernel, 1, sizeof(cl_mem), &divI);
                err |= clSetKernelArg(vWKStressKernel, 2, sizeof(cl_mem), &divJ);
                if(err != CL_SUCCESS){
                    cerr << "ERROR in setting a vW stress kernel argument: " << err << endl;
                    throw runtime_error("Failed to set a vW stress kernel argument.");
                }
        
                // enqueue the kernel over the entire "1D'd" grid
                const size_t realPoints = workGrid->getTotalGridPoints();
                err = clEnqueueNDRangeKernel(_queue, vWKStressKernel, 1, NULL, &realPoints, NULL, 0, NULL, NULL);
                if (err != CL_SUCCESS){
                    cerr << "ERROR: 1D-enqueue of vW stress kernel failed. " << err << endl;
                    throw runtime_error("1D-enqueue of vW stress kernel failed.");
                }
                
                // sum over the working variable
                const double sum = workGrid->sumOver();

                // assemble
                tensor->at(i,j) = -sum/totalPoints;

                // symmetrize
                tensor->at(j,i) = tensor->at(i,j);
            }
        }

        return stress;
    }
    
private:
    
    cl_context _ctx;
    cl_command_queue _queue;
    
    cl_program vWOCLProg;
    
    cl_kernel vWKPotKernel;
    size_t _localvWPotSize;
    
    cl_kernel vWKStressKernel;
    size_t _localvWStressSize;
    
    void setupKernels(CartesianOCLOOPGrid* example){
        
        _ctx = example->getGPUContext();
        clRetainContext(_ctx);
        _queue = example->getGPUQueue();
        clRetainCommandQueue(_queue);
        cl_device_id* devices = example->getGPUDevices();
        cl_uint noDevices = example->getNoGPUDevices();

        const string st1 =                                      "\n" \
"__kernel void vwPot(  __global const double *density,           \n" \
"    __global double *potential, __global const double *laplacian){\n" \
"    const size_t idx = get_global_id(0);                        \n" \
"    const double d = density[idx];                              \n" \
"    const double tmp = laplacian[idx]/d;                        \n" \
"    const double pre = (d < 1E-6) ? 0.0 : -0.5*tmp;             \n" \
"    potential[idx] = pre;                                       \n" \
"}                                                               \n" \
"__kernel void vwStress(  __global double *grid,                 \n" \
"    __global const double *divI, __global const double *divJ){  \n" \
"    const size_t idx = get_global_id(0);                        \n" \
"    grid[idx] = divI[idx]*divJ[idx];                            \n" \
"}                                                               \n" \
                                                                "\n" ;
        
        const string macros = example->getMacroDefinitions();
    
        ostringstream os;
        os << macros;
        os << st1;

        const string s = os.str();
        const char* vWOCLT = s.c_str();
        
        cl_int err;
        this->vWOCLProg = clCreateProgramWithSource(_ctx, 1, (const char**) &vWOCLT, NULL, &err);
        if(!vWOCLProg || err != CL_SUCCESS){
            cerr << "ERROR to create vW OCL program " << err << endl;
            throw runtime_error("Failed to create vW OCL program.");
        }
        err = clBuildProgram(vWOCLProg, noDevices, devices, example->getCompilationOptions(), NULL, NULL);
        if(err != CL_SUCCESS){
            cerr << "ERROR in building vW OCL program " << err << endl;
            
            cl_build_status status;
            // check build error and build status first
            clGetProgramBuildInfo(vWOCLProg, devices[0], CL_PROGRAM_BUILD_STATUS, 
                sizeof(cl_build_status), &status, NULL);
 
            if(!(err == CL_BUILD_PROGRAM_FAILURE && status == 0)){ // this is what I observe currently on NVIDIA and it seems to be caused by the KEDFOCL macro, makes no sense
            
                // check build log
                size_t logSize;
                clGetProgramBuildInfo(vWOCLProg, devices[0], 
                        CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);
                char* programLog = (char*) calloc (logSize+1, sizeof(char));
                clGetProgramBuildInfo(vWOCLProg, devices[0], 
                        CL_PROGRAM_BUILD_LOG, logSize+1, programLog, NULL);
                printf("Build failed; error=%d, status=%d, programLog:nn%s \n", 
                        err, status, programLog);
                free(programLog);
        
                throw runtime_error("Could not build vW OCL program.");
            }
        }
        
        this->vWKPotKernel = clCreateKernel(vWOCLProg, "vwPot", &err);
        if(err != CL_SUCCESS){
            cerr << "ERROR in creating vW potential kernel " << err << endl;
            throw runtime_error("Could not create vW potential kernel.");
        }
        
        err = clGetKernelWorkGroupInfo(vWKPotKernel, devices[0], CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &_localvWPotSize, NULL);
        if (err != CL_SUCCESS){
            cerr << "ERROR to inquire local work group size for vW potential kernel " << err << endl;
            throw runtime_error("Not possible to inquire local work group size of vW potential kernel.");
        }

        
        this->vWKStressKernel = clCreateKernel(vWOCLProg, "vwStress", &err);
        if(err != CL_SUCCESS){
            cerr << "ERROR in creating vW stress kernel " << err << endl;
            throw runtime_error("Could not create vW stress kernel.");
        }
        
        err = clGetKernelWorkGroupInfo(vWKStressKernel, devices[0], CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &_localvWStressSize, NULL);
        if (err != CL_SUCCESS){
            cerr << "ERROR to inquire local work group size for vW stress kernel " << err << endl;
            throw runtime_error("Not possible to inquire local work group size of vW stress kernel.");
        }
    }
};


#endif /* VONWEIZSAECKEROCL_HPP */

