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

#ifndef THOMASFERMIOCL_HPP
#define THOMASFERMIOCL_HPP

#include <CL/cl.h>
#include <string>
#include "KEDF.hpp"
#include "KEDFConstants.hpp"
#include "CartesianOCLOOPGrid.hpp"
using namespace arma;
using namespace std;

template<>
class ThomasFermi<CartesianOCLOOPGrid>: public KEDF<CartesianOCLOOPGrid> {
public:
    ThomasFermi(CartesianOCLOOPGrid* example){
        setupKernels(example);
    }
    
    ~ThomasFermi(){
        clReleaseCommandQueue(_queue);
        clReleaseContext(_ctx);
        clReleaseProgram(tfCLProg);
        clReleaseKernel(pow53Kernel);
        clReleaseKernel(pow53PotKernel);
    }
    
    string getMethodDescription() const {
        return "Thomas-Fermi KEDF (OpenCL version)";
    }
    
    vector<string> getCitations() const {
        
        vector<string> citations(0);
        citations.push_back("L. H. Thomas, Proc. Cambridge Philos. Soc. 23, 542 (1927).");
        citations.push_back("E. Fermi, Rend. Accad. Naz. Lincei 6, 602 (1927).");
        citations.push_back("E. Fermi, Z. Phys. 48, 73 (1928).");

        return citations;
    }
    
    vector<string> getWorkingEquations() const {
        
        vector<string> equations(0);
        equations.push_back("e = cTF \\cdot \\sum \\rho^{\\frac{5}{3}}");
    
        return equations;
    }
    
    double calcEnergy(const CartesianOCLOOPGrid& grid) const {
        
        unique_ptr<CartesianOCLOOPGrid> workGrid(grid.duplicate());
        
        // transfer the real grid to the GPU
        workGrid->transferRealToGPU();
        
        // get the buffer and hook it into the kernel
        cl_mem buff = workGrid->getRealGPUBuffer();
        
        cl_int err;
        err = clSetKernelArg(pow53Kernel, 0, sizeof(cl_mem), &buff);
        if(err != CL_SUCCESS){
            cerr << "ERROR in setting pow53 kernel argument 0: " << err << endl;
            throw runtime_error("Failed to set pow53 kernel argument 0.");
        }
        
        // enqueue the kernel over the entire "1D'd" grid
        const size_t points = workGrid->getTotalGridPoints()/grid.getVectortypeAlignment();
        err = clEnqueueNDRangeKernel(_queue, pow53Kernel, 1, NULL, &points, NULL, 0, NULL, NULL);
        if (err != CL_SUCCESS){
            printf("ERROR enqueing pow53: %d\n",err);
            throw runtime_error("1D-enqueue of pow53 kernel failed.");
        }
        
        workGrid->markRealDirty();
        workGrid->markReciDirty();
        
        const double energy = workGrid->integrate();
    
        return _CTF*energy;
    }
    
    double calcPotential(const CartesianOCLOOPGrid& grid, CartesianOCLOOPGrid& potential) const {
        
        unique_ptr<CartesianOCLOOPGrid> workGrid(grid.duplicate());
        
        // transfer the real density grid to the GPU
        workGrid->transferRealToGPU();
        
        // get the buffers and hook them into the kernel
        cl_mem buff = workGrid->getRealGPUBuffer();
        cl_mem pot = potential.getRealGPUBuffer();
        
        potential.markRealDirty();
        potential.markReciDirty();
        potential.markRealGPUClean();
        potential.markReciGPUDirty();
        
        cl_int err;
        err  = clSetKernelArg(pow53PotKernel, 0, sizeof(cl_mem), &buff);
        err |= clSetKernelArg(pow53PotKernel, 1, sizeof(cl_mem), &pot);
        if(err != CL_SUCCESS){
            cerr << "ERROR in setting a pow53Pot kernel argument: " << err << endl;
            throw runtime_error("Failed to set a pow53Pot kernel argument.");
        }
        
        // enqueue the kernel over the entire "1D'd" grid
        const size_t realPoints = workGrid->getTotalGridPoints()/grid.getVectortypeAlignment();
        err = clEnqueueNDRangeKernel(_queue, pow53PotKernel, 1, NULL, &realPoints, NULL, 0, NULL, NULL);
        if (err != CL_SUCCESS){
            throw runtime_error("1D-enqueue of pow53 potential kernel failed.");
        }
        
        workGrid->markRealDirty();
        workGrid->markReciDirty();
    
        const double energy = workGrid->integrate();
    
        return _CTF*energy;
    }
    
    unique_ptr<StressTensor> calcStress(const CartesianOCLOOPGrid& grid) const {
        
        unique_ptr<StressTensor> stress = make_unique<StressTensor>();
        mat* const tensor = stress->getTensor();

        const double energy = this->calcEnergy(grid);

        const double norm = grid.stressNorm();
        const double stressElem = -2.0*energy/norm;

        tensor->at(0,0) = stressElem;
        tensor->at(1,1) = stressElem;
        tensor->at(2,2) = stressElem;

        return stress;
    }
    
private:
    
    const double _CTF = 2.87123400018819;
    
    cl_context _ctx;
    cl_command_queue _queue;
    
    cl_program tfCLProg;
    
    cl_kernel pow53Kernel;
    size_t _localPow53Size;
    
    cl_kernel pow53PotKernel;
    size_t _localPow53PotSize;
    
    void setupKernels(CartesianOCLOOPGrid* example){
        
        _ctx = example->getGPUContext();
        clRetainContext(_ctx);
        _queue = example->getGPUQueue();
        clRetainCommandQueue(_queue);
        cl_device_id* devices = example->getGPUDevices();
        cl_uint noDevices = example->getNoGPUDevices();
        
        const string st1 =                                      "\n" \
"__kernel void pow53Grid(  __global KEDFOCLV *data){             \n" \
"    const size_t idx = get_global_id(0);                        \n" \
"    const KEDFOCLV d = data[idx];                               \n" \
"    const KEDFOCLV dSq = d*d;                                   \n" \
"    const KEDFOCLV d23 = cbrt(dSq);                             \n" \
"    data[idx] = d23*d;                                          \n" \
"}                                                               \n" \
"__kernel void pow53PotGrid(  __global KEDFOCLV *data,           \n" \
"                             __global KEDFOCLV *pot){           \n" \
"    const size_t idx = get_global_id(0);                        \n" \
"    const KEDFOCLV d = data[idx];                               \n" \
"    const KEDFOCLV dSq = d*d;                                   \n" \
"    const KEDFOCLV d23 = cbrt(dSq);                             \n" \
"    const KEDFOCLV d53 = d23*d;                                 \n" \
"    const double pre = 5.0/3.0*2.87123400018819;                \n" \
"    data[idx] = d53;                                            \n" \
"    pot[idx] = pre*d23;                                         \n" \
"}                                                               \n" \
                                                                "\n" ;
        
        const string macros = example->getMacroDefinitions();
    
        ostringstream os;
        os << macros;
        os << st1;

        const string s = os.str();    
        const char* tfCL = s.c_str();
        
        cl_int err;
        this->tfCLProg = clCreateProgramWithSource(_ctx, 1, (const char**) &tfCL, NULL, &err);
        if(!tfCLProg || err != CL_SUCCESS){
            cerr << "ERROR to create TF OCL program " << err << endl;
            throw runtime_error("Failed to create TF OCL program.");
        }
        err = clBuildProgram(tfCLProg, noDevices, devices, example->getCompilationOptions(), NULL, NULL);
        if(err != CL_SUCCESS){
            cerr << "ERROR in building TF OCL program " << err << endl;
            
            cl_build_status status;
            // check build error and build status first
            clGetProgramBuildInfo(tfCLProg, devices[0], CL_PROGRAM_BUILD_STATUS, 
                sizeof(cl_build_status), &status, NULL);
            
            if(!(err == CL_BUILD_PROGRAM_FAILURE && status == 0)){ // this is what I observe currently on NVIDIA and it seems to be caused by the KEDFOCL macro, makes no sense
 
                // check build log
                size_t logSize;
                clGetProgramBuildInfo(tfCLProg, devices[0], 
                    CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);
                char* programLog = (char*) calloc (logSize+1, sizeof(char));
                clGetProgramBuildInfo(tfCLProg, devices[0], 
                        CL_PROGRAM_BUILD_LOG, logSize+1, programLog, NULL);
                printf("Build failed; error=%d, status=%d, programLog:nn%s \n", 
                        err, status, programLog);
                free(programLog);
        
                throw runtime_error("Could not build TF OCL program.");
            }
        }
        
        this->pow53Kernel = clCreateKernel(tfCLProg, "pow53Grid", &err);
        if(err != CL_SUCCESS){
            cerr << "ERROR in creating pow53 kernel " << err << endl;
            throw runtime_error("Could not create pow53 kernel.");
        }
        
        err = clGetKernelWorkGroupInfo(pow53Kernel, devices[0], CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &_localPow53Size, NULL);
        if (err != CL_SUCCESS){
            cerr << "ERROR to inquire local work group size for pow53 kernel " << err << endl;
            throw runtime_error("Not possible to inquire local work group size of pow53 kernel.");
        }
               
        this->pow53PotKernel = clCreateKernel(tfCLProg, "pow53PotGrid", &err);
        if(err != CL_SUCCESS){
            cerr << "ERROR in creating pow53Pot kernel " << err << endl;        
            throw runtime_error("Could not create pow53Pot kernel.");
        }
        
        err = clGetKernelWorkGroupInfo(pow53PotKernel, devices[0], CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &_localPow53PotSize, NULL);
        if (err != CL_SUCCESS){
            cerr << "ERROR to inquire local work group size for pow53Pot kernel " << err << endl;
            throw runtime_error("Not possible to inquire local work group size of pow53Pot kernel.");
        }
    }
};

#endif /* THOMASFERMIOCL_HPP */

