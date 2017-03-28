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
#include "CartesianOCLOOPGrid.hpp"
using namespace arma;

#define MAXPLATFORMS 16
#define MAXDEVICES 16

//const string CartesianOCLOOPGrid::COMPILATIONOPTS = "-Werror -cl-std=CL1.1";
//const string CartesianOCLOOPGrid::COMPILATIONOPTS = "-Werror -cl-no-signed-zeros";
const string CartesianOCLOOPGrid::COMPILATIONOPTS = "-Werror -cl-std=CL1.2";

CartesianOCLOOPGrid::CartesianOCLOOPGrid(const size_t xDim, const size_t yDim, const size_t zDim, const shared_ptr<mat> cellVectors,
        const size_t platformNo, const size_t deviceNo)
: CartesianGrid(xDim,yDim,zDim,cellVectors){
    
    this->_realInSync = true;
    this->_reciInSync = true;
    this->_realReturned = true;
    this->_reciReturned = true;
    this->_realOnGPUInSync = false;
    this->_reciOnGPUInSync = false;
    
    // do lazy allocation of GPU memory to avoid having too many grids in GPU memory at the same time
    // this will obviously come at the expense of overhead but we can be smart when we need to compute
    // a more advanced quantity (say the divergence/gradient)
    _bufReal = NULL;
    _bufReci = NULL;
    
    cl_int err;
    const clfftDim dim = CLFFT_3D;
    size_t* clLengths = new size_t[3];
    clLengths[0] = _xDim; // our memory layout is col-major
    clLengths[1] = _yDim;
    clLengths[2] = _zDim;
    
    // allocate host memory buffer which can in turn be used by arma cube objects
    this->_bufferSizeHost  = 2*_zDim*_yDim*(_xDim/2+1)*sizeof(double);
#ifdef LIBKEDF_NO_ALIGNED_MEMORY
    _hostMem = malloc(_bufferSizeHost);
#else
    // AVX wants at least 16, but rather 32 byte alignment for some features. This is basically 4 doubles in one line
    // upcoming AVX-512 operates on 64 (!) bytes of data simultaneously
    const size_t alignment = LIBKEDF_ALIGNMENT;
    const int error = posix_memalign(&_hostMem,alignment,_bufferSizeHost);
    if(error != 0){
        cerr << "ERROR: Can't get aligned host memory: " << error << ", size: " << _bufferSizeHost << ", alignment: " << alignment << endl;
        throw runtime_error("No aligned memory could be allocated.");
    }
#endif
    
    // use memory for arma backends. we define that the real storage is fully compressed (i.e., whatever extra bytes are in the end)
    this->_realGrid = new cube((double*)_hostMem,_xDim,_yDim,_zDim,false,true);
    this->_reciGrid = new cx_cube((complex<double>*)_hostMem,_xRecDim,_yRecDim,_zRecDim,false,true);
    
    cl_uint num_entries = MAXPLATFORMS; // maximum MAXPLATFORMS platforms entries
    _noPlatforms = -1;
    cl_platform_id* platforms = new cl_platform_id[num_entries];
    
    err = clGetPlatformIDs(num_entries, platforms, &_noPlatforms);
    if(err != CL_SUCCESS){
        cerr << "ERROR in getting platform IDs: " << err << endl;
        throw runtime_error("Failure to get platform ID.");
    }
    
    if(_noPlatforms <= 0){
        cerr << "ERROR: No OpenCL platform found! " << _noPlatforms << endl;
        throw runtime_error("No OpenCL platform found.");
    } else if(_noPlatforms <= platformNo){
        cerr << "ERROR: Less platforms than required found! " << _noPlatforms << " vs " << platformNo << endl;
        throw runtime_error("Less platforms found than required to use configured ID.");
    }
    this->_platforms.reset(platforms, default_delete<cl_platform_id[]>());
    
    cl_uint num_devices = -1;
    cl_platform_id platform = platforms[platformNo];    
    cl_device_id devs[MAXDEVICES];
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, MAXDEVICES, devs, &num_devices);
    if(err != CL_SUCCESS){
        printf("ERROR in getting device IDs: %d\n",err);
        fflush(stdout);
        exit(err);
    }

    cl_device_id* ctx_dev = (cl_device_id*) calloc(1,sizeof(cl_device_id));
    memcpy ( ctx_dev, &devs[deviceNo], sizeof(cl_device_id) );

    this->_devices.reset(ctx_dev, default_delete<cl_device_id[]>());
    this->_noDevices = 1; // force one for the time being

    if(num_devices <= 0) {
        cerr << "ERROR: No OpenCL devices found! " << num_devices << endl;
        throw runtime_error("No OpenCL devices found.");
    } else if(num_devices <= deviceNo){
        cerr << "ERROR: Less devices than required found! " << num_devices << " vs " << deviceNo << endl;
        throw runtime_error("Less devices found than required to use configured ID.");
    }
    
    cl_context_properties* props = (cl_context_properties*) malloc(3*sizeof(cl_context_properties));
    props[0] = CL_CONTEXT_PLATFORM;
    props[1] = (cl_context_properties) platform;
    props[2] = 0;
    this->_ctx = clCreateContext(props, 1, ctx_dev, NULL, NULL, &err );

    cerr << "Context: " << _ctx << endl;
    if(err != CL_SUCCESS){
        printf("ERROR in creating context: %d\n",err);
        fflush(stdout);
        exit(err);
    }
    this->_queue = clCreateCommandQueue( _ctx, _devices.get()[0], 0, &err );
    if(err != CL_SUCCESS){
        printf("ERROR in creating queue: %d\n",err);
        fflush(stdout);
        exit(err);
    }
    
    clfftPlanHandle* r2c_plan = (clfftPlanHandle*) malloc(sizeof(clfftPlanHandle));
    err = clfftCreateDefaultPlan(r2c_plan, _ctx, dim, clLengths);
    if(err != CL_SUCCESS){
        cerr << "ERROR in creating R2C plan: " << err << "\tdims: " << dim << "\t clLenghts: " << clLengths[0] << "\t" << clLengths[1] << "\t" << clLengths[2] << endl;
        throw runtime_error("Could not create R2C plan.");
    }
    
    /* set the plan up */
    err = clfftSetPlanPrecision(*r2c_plan, CLFFT_DOUBLE);
    err |= clfftSetLayout(*r2c_plan, CLFFT_REAL, CLFFT_HERMITIAN_INTERLEAVED);
    err |= clfftSetResultLocation(*r2c_plan, CLFFT_OUTOFPLACE);
    
    /* fashion the strides for dense packing of numbers */
    size_t clInStridesR2C[4];
    clInStridesR2C[0] = 1;
    clInStridesR2C[1] = clInStridesR2C[0]*clLengths[0];
    clInStridesR2C[2] = clInStridesR2C[1]*clLengths[1];
    clInStridesR2C[3] = clInStridesR2C[2]*clLengths[2];

    size_t clOutStridesR2C[4];
    clOutStridesR2C[0] = 1;
    clOutStridesR2C[1] = clOutStridesR2C[0]*(clLengths[0]/2+1);
    clOutStridesR2C[2] = clOutStridesR2C[1]*clLengths[1];
    clOutStridesR2C[3] = clOutStridesR2C[2]*clLengths[2];

    err |= clfftSetPlanInStride(*r2c_plan,dim,clInStridesR2C);
    err |= clfftSetPlanOutStride(*r2c_plan,dim,clOutStridesR2C);
    err |= clfftSetPlanDistance(*r2c_plan,clInStridesR2C[3],clOutStridesR2C[3]);
    cl_float scaleFWD = 1.0/_noGridPoints;
    cl_float scaleBWD = 1.0;
    err |= clfftSetPlanScale(*r2c_plan,CLFFT_FORWARD,scaleFWD);
    err |= clfftSetPlanScale(*r2c_plan,CLFFT_BACKWARD,scaleBWD);
    
    if(err != CL_SUCCESS){
      cerr << "ERROR to create r2c clFFT plan. " << err << endl;
      throw runtime_error("Failure to create r2c clFFT plan.");
    }

    /* c2r plan*/
    clfftPlanHandle* c2r_plan = (clfftPlanHandle*) malloc(sizeof(clfftPlanHandle));
    err = clfftCreateDefaultPlan(c2r_plan, _ctx, dim, clLengths); // XXX clLengths must be adapted, I bet!
    
    /* set the plan up */
    err |= clfftSetPlanPrecision(*c2r_plan, CLFFT_DOUBLE);
    err |= clfftSetLayout(*c2r_plan, CLFFT_HERMITIAN_INTERLEAVED, CLFFT_REAL);
    err |= clfftSetResultLocation(*c2r_plan, CLFFT_OUTOFPLACE);

    /* the c2r strides are just the r2c strides the other way round*/
    err |= clfftSetPlanInStride(*c2r_plan,dim,clOutStridesR2C);
    err |= clfftSetPlanOutStride(*c2r_plan,dim,clInStridesR2C);
    err |= clfftSetPlanDistance(*r2c_plan,clOutStridesR2C[3],clInStridesR2C[3]);
    err |= clfftSetPlanScale(*c2r_plan,CLFFT_FORWARD,scaleFWD);
    err |= clfftSetPlanScale(*c2r_plan,CLFFT_BACKWARD,scaleBWD);

    if(err != CL_SUCCESS){
      cerr << "ERROR to create c2r clFFT plan. " << err << endl;
      throw runtime_error("Failure to create c2r clFFT plan.");
    }
    
    /* Bake the plans */
    err = clfftBakePlan(*r2c_plan, 1, &_queue, NULL, NULL);
    err = clfftBakePlan(*c2r_plan, 1, &_queue, NULL, NULL);
    
    this->_r2c_plan.reset(r2c_plan);
    this->_c2r_plan.reset(c2r_plan);
    
    // now take care of compiling some other kernels
    const string st1 =                                          "\n" \
"__kernel void sqrtGrid(  __global KEDFOCLV *data){              \n" \
"    const size_t idx = get_global_id(0);                        \n" \
"    data[idx] = sqrt(data[idx]);                                \n" \
"}                                                               \n" \
"__kernel void powGrid(__global KEDFOCLV *data,                  \n" \
"                      const double exponent){                   \n" \
"    const size_t idx = get_global_id(0);                        \n" \
"    data[idx] = pow(data[idx], exponent);                       \n" \
"}                                                               \n" \
"__kernel void addGrid(__global KEDFOCLV *data,                  \n" \
"                      __global const KEDFOCLV *other){          \n" \
"    const size_t idx = get_global_id(0);                        \n" \
"    data[idx] += other[idx];                                    \n" \
"}                                                               \n" \
"__kernel void fmaGrid(__global double *data,                    \n" \
"                      __global const double *other,             \n" \
"                      const double multiplier){                 \n" \
"    const size_t idx = get_global_id(0);                        \n" \
"    fma(multiplier, other[idx], data[idx]);                     \n" \
"}                                                               \n" \
"__kernel void realMultGrid(__global KEDFOCLV *data,             \n" \
"                 __global const KEDFOCLV *multipliers){         \n" \
"    const size_t idx = get_global_id(0);                        \n" \
"    data[idx] *= multipliers[idx];                              \n" \
"}                                                               \n" \
"__kernel void mult2SqrtGrid(__global KEDFOCLV *data,            \n" \
"                 __global const KEDFOCLV *other){               \n" \
"    const size_t idx = get_global_id(0);                        \n" \
"    data[idx] *= 2*sqrt(other[idx]);                            \n" \
"}                                                               \n" \
"__kernel void addRealSqGrid(__global KEDFOCLV *data,            \n" \
"                 __global const KEDFOCLV *other){               \n" \
"    const size_t idx = get_global_id(0);                        \n" \
"    data[idx] += other[idx]*other[idx];                         \n" \
"}                                                               \n" \
"__kernel void addRealSq0Grid(__global KEDFOCLV *data,           \n" \
"                 __global const KEDFOCLV *other){               \n" \
"    const size_t idx = get_global_id(0);                        \n" \
"    data[idx] = other[idx]*other[idx];                          \n" \
"}                                                               \n" \
"__kernel void cmplMultGrid(__global double *data,               \n" \
"                 __global const double *multipliers){           \n" \
"    const size_t idx = get_global_id(0);                        \n" \
"    const double re1 = data[2*idx];                             \n" \
"    const double im1 = data[2*idx+1];                           \n" \
"    const double re2 = multipliers[2*idx];                      \n" \
"    const double im2 = multipliers[2*idx+1];                    \n" \
"    data[2*idx] = re1*re2 - im1*im2;                            \n" \
"    data[2*idx+1] = re1*im2 + re2*im1;                          \n" \
"}                                                               \n" \
"__kernel void cmplRealMultGrid(__global double *data,           \n" \
"                 __global const double *multipliers){           \n" \
"    const size_t idx = get_global_id(0);                        \n" \
"    const double re2 = multipliers[idx];                        \n" \
"    data[2*idx] *= re2;                                         \n" \
"    data[2*idx+1] *= re2;                                       \n" \
"}                                                               \n" \
"__kernel void pureCmplMultGrid(__global double *data,           \n" \
"                 __global const double *multipliers){           \n" \
"    const size_t idx = get_global_id(0);                        \n" \
"    const double re1 = data[2*idx];                             \n" \
"    const double im1 = data[2*idx+1];                           \n" \
"    const double im2 = multipliers[idx];                        \n" \
"    data[2*idx] = - im1*im2;                                    \n" \
"    data[2*idx+1] = im2*re1;                                    \n" \
"}                                                               \n" \
"__kernel void gNormsMultGrid(__global double2 *data,            \n" \
"                 __global const double *gNorms){                \n" \
"    const size_t idx = get_global_id(0);                        \n" \
"    const double tmp = -1*gNorms[idx]*gNorms[idx];              \n" \
"    data[idx] *= tmp;                                           \n" \
"}                                                               \n" \
"__kernel void addUp(__global double4* data,                     \n" \
"  __local double* localResult, __global double* groupResult) {  \n" \
"    const size_t idx = get_global_id(0) * 2;                    \n" \
"    const double4 inp1 = data[idx];                             \n" \
"    const double4 inp2 = data[idx+1];                           \n" \
"    const double4 sumVec = inp1+inp2;                           \n" \
"    const size_t locID = get_local_id(0);                       \n" \
"    localResult[locID] = sumVec.s0 + sumVec.s1 +                \n" \
"                         sumVec.s2 + sumVec.s3;                 \n" \
"    barrier(CLK_LOCAL_MEM_FENCE);                               \n" \
"    if(locID == 0) {                                            \n" \
"       double sum = 0.0;                                        \n" \
"       for(int i = 0; i < get_local_size(0); i++)               \n" \
"           sum += localResult[i];                               \n" \
"       groupResult[get_group_id(0)] = sum;                      \n" \
"    }                                                           \n" \
"}                                                               \n" \
"__kernel void minMaxFast(__global double4* data,                \n" \
"  __local double* localMinResult, __global double* groupMinResult, \n" \
"  __local double* localMaxResult, __global double* groupMaxResult){ \n" \
"    const size_t idx = get_global_id(0) * 2;                    \n" \
"    const double4 inp1 = data[idx];                             \n" \
"    const double4 inp2 = data[idx+1];                           \n" \
"    const double4 minVec = min(inp1,inp2);                      \n" \
"    const double4 maxVec = max(inp1,inp2);                      \n" \
"    const size_t locID = get_local_id(0);                       \n" \
"    const double locMin1 = min(minVec.s0, minVec.s1);           \n" \
"    const double locMin2 = min(minVec.s2, minVec.s3);           \n" \
"    const double locMax1 = max(maxVec.s0, maxVec.s1);           \n" \
"    const double locMax2 = max(maxVec.s2, maxVec.s3);           \n" \
"    localMinResult[locID] = min(locMin1,locMin2);               \n" \
"    localMaxResult[locID] = max(locMax1,locMax2);               \n" \
"    barrier(CLK_LOCAL_MEM_FENCE);                               \n" \
"    if(locID == 0) {                                            \n" \
"       double minRes = localMinResult[0];                       \n" \
"       double maxRes = localMaxResult[0];                       \n" \
"       for(int i = 1; i < get_local_size(0); i++) {             \n" \
"           minRes = min(minRes,localMinResult[i]);              \n" \
"           maxRes = max(maxRes,localMaxResult[i]);              \n" \
"       }                                                        \n" \
"       groupMinResult[get_group_id(0)] = minRes;                \n" \
"       groupMaxResult[get_group_id(0)] = maxRes;                \n" \
"    }                                                           \n" \
"}                                                               \n" \
"__kernel void sumOverGrid(__global const double *data,          \n" \
"         __global double *res, const int sliceLength){          \n" \
"    const size_t idx = get_global_id(0);                        \n" \
"    double sum = 0.0;                                           \n" \
"    for(size_t x = sliceLength*idx; x < sliceLength*(idx+1); ++x){  \n" \
"        sum += data[x];                                         \n" \
"    }                                                           \n" \
"    res[idx] = sum;                                             \n" \
"}                                                               \n" \
"__kernel void minMaxGrid(__global const double *data,           \n" \
"         __global double *minRes, __global double *maxRes,      \n" \
"         const int sliceLength){                                \n" \
"    const size_t idx = get_global_id(0);                        \n" \
"    double myMin = 1e42;                                        \n" \
"    double myMax = 0.0;                                         \n" \
"    for(size_t x = sliceLength*idx; x < sliceLength*(idx+1); ++x){  \n" \
"        myMin = min(myMin,data[x]);                             \n" \
"        myMax = max(myMax,data[x]);                             \n" \
"    }                                                           \n" \
"    minRes[idx] = myMin;                                        \n" \
"    maxRes[idx] = myMax;                                        \n" \
"}                                                               \n" \
                                                                "\n" ;
    
#ifdef LIBKEDF_HAS_PRINTF
    const string st2 =                                          "\n" \
"__kernel void realPrint(__global const double *data){           \n" \
"    size_t idx = get_global_id(0);                              \n" \
"    printf(\" %d %10.6f  \",idx,data[idx]);                     \n" \
"}                                                               \n" \
"__kernel void complexPrint(__global const double *data){        \n" \
"    const size_t idx = get_global_id(0);                        \n" \
"    printf(\"%d (%10.6f, %10.6f)  \",idx,data[2*idx],data[2*idx+1]);\n" \
"}                                                               \n" \
                                                                "\n" ;
#else
    cerr << "WARNING: No printf for OpenCL kernels, hence no debug output!" << endl;
        const string st2 =                                      "\n" \
"__kernel void realPrint(__global const float *data){            \n" \
"}                                                               \n" \
"__kernel void complexPrint(__global const float *data){         \n" \
"}                                                               \n" \
                                                                "\n" ;
#endif
    
    const string macros = getMacroDefinitions();
    
    ostringstream os;
    os << macros;
    os << st1;
    os << "\n";
    os << st2;
   
    const string s = os.str();
    const char* st = s.c_str();
    
    this->cartOCLOOCPProgram = clCreateProgramWithSource(_ctx, 1, (const char**) &st, NULL, &err);
    if(!cartOCLOOCPProgram || err != CL_SUCCESS){
        cerr << "ERROR to create Cartesian OOP OCL program " << err << endl;
        throw runtime_error("Failed to create Cartesian OOP OCL program.");
    }
    const char* compOpts = getCompilationOptions();
    printf("%s\n",compOpts);
    err = clBuildProgram(cartOCLOOCPProgram, 1, _devices.get(), compOpts, NULL, NULL);
    if(err != CL_SUCCESS){
        cerr << "ERROR in building Cartesian OOP OCL program " << err << endl;
        
        cl_build_status status;
        // check build error and build status first
        clGetProgramBuildInfo(cartOCLOOCPProgram, *ctx_dev, CL_PROGRAM_BUILD_STATUS, 
                sizeof(cl_build_status), &status, NULL);
 
        if(!(err == CL_BUILD_PROGRAM_FAILURE && status == 0)){ // this is what I observe currently on NVIDIA and it seems to be caused by the KEDFOCL macro, makes no sense
            // check build log
            size_t logSize;
            clGetProgramBuildInfo(cartOCLOOCPProgram, *ctx_dev,
                    CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);
            char* programLog = (char*) calloc (logSize+1, sizeof(char));
            clGetProgramBuildInfo(cartOCLOOCPProgram, *ctx_dev,
                    CL_PROGRAM_BUILD_LOG, logSize+1, programLog, NULL);
            printf("Build failed; error=%d, status=%d, programLog:nn%s \n", 
                    error, status, programLog);
            free(programLog);
        
            throw runtime_error("Could not build Cartesian OOP OCL  program.");
        }
    }
    
    this->sqrtKernel = clCreateKernel(cartOCLOOCPProgram, "sqrtGrid", &err);
    if (!sqrtKernel || err != CL_SUCCESS){
        cerr << "ERROR to create sqrt kernel " << err << endl;
        throw runtime_error("Failed to create sqrt kernel.");
    }
    
    err = clGetKernelWorkGroupInfo(sqrtKernel, *ctx_dev, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &_localSqrtSize, NULL);
    if (err != CL_SUCCESS){
        cerr << "ERROR to inquire local work group size for sqrt kernel " << err << endl;
        throw runtime_error("Not possible to inquire local work group size of sqrt kernel.");
    }
    
    // SETUP THE POW KERNEL
    this->powKernel = clCreateKernel(cartOCLOOCPProgram, "powGrid", &err);
    if (!powKernel || err != CL_SUCCESS){
        cerr << "ERROR to create pow kernel " << err << endl;        
        throw runtime_error("Failed to create pow kernel.");
    }
    
    err = clGetKernelWorkGroupInfo(powKernel, *ctx_dev, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &_localPowSize, NULL);
    if (err != CL_SUCCESS){
        cerr << "ERROR to inquire local work group size for pow kernel " << err << endl;
        throw runtime_error("Not possible to inquire local work group size of pow kernel.");
    }
    
    // SETUP THE ADD KERNEL
    this->addKernel = clCreateKernel(cartOCLOOCPProgram, "addGrid", &err);
    if (!addKernel || err != CL_SUCCESS){
        cerr << "ERROR to create add kernel " << err << endl;        
        throw runtime_error("Failed to create add kernel.");
    }
    
    // SETUP THE FMA KERNEL
    this->fmaKernel = clCreateKernel(cartOCLOOCPProgram, "fmaGrid", &err);
    if (!fmaKernel || err != CL_SUCCESS){
        cerr << "ERROR to create fma kernel " << err << endl;        
        throw runtime_error("Failed to create fma kernel.");
    }
        
    // SETUP THE POINTWISE REAL MULTIPLICATION KERNEL
    this->realMultKernel = clCreateKernel(cartOCLOOCPProgram, "realMultGrid", &err);
    if (!realMultKernel || err != CL_SUCCESS){
        cerr << "ERROR to create real multiplication kernel " << err << endl;
        throw runtime_error("Failed to create real multiplication kernel.");
    }
    
    err = clGetKernelWorkGroupInfo(realMultKernel, *ctx_dev, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &_localRealMultSize, NULL);
    if (err != CL_SUCCESS){
        cerr << "ERROR to inquire local work group size for real multiplication kernel " << err << endl;
        throw runtime_error("Not possible to inquire local work group size of real multiplication kernel.");
    }
    
    // SETUP THE POINTWISE REAL MULTIPLICATION WITH THE 2*SQRT OF ANOTHER GRID
    this->mult2SqrtKernel = clCreateKernel(cartOCLOOCPProgram, "mult2SqrtGrid", &err);
    if (!mult2SqrtKernel || err != CL_SUCCESS){
        cerr << "ERROR to create real multiplication with two sqrt kernel " << err << endl;
        throw runtime_error("Failed to create real multiplication with two sqrt kernel.");
    }
    
    // SETUP THE POINTWISE REAL ADDITION OF A SQUARED GRID KERNEL
    this->realAddSqKernel = clCreateKernel(cartOCLOOCPProgram, "addRealSqGrid", &err);
    if (!realAddSqKernel || err != CL_SUCCESS){
        cerr << "ERROR to create real addition of squared grid kernel " << err << endl;
        throw runtime_error("Failed to create real addition of squared grid  kernel.");
    }
    
    err = clGetKernelWorkGroupInfo(realAddSqKernel, *ctx_dev, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &_localRealAddSqSize, NULL);
    if (err != CL_SUCCESS){
        cerr << "ERROR to inquire local work group size for real addition of squared grid kernel " << err << endl;
        throw runtime_error("Not possible to inquire local work group size of real addition of squared grid kernel.");
    }
    
    // SETUP THE POINTWISE REAL ADDITION OF A SQUARED GRID KERNEL
    this->realAddSq0Kernel = clCreateKernel(cartOCLOOCPProgram, "addRealSq0Grid", &err);
    if (!realAddSq0Kernel || err != CL_SUCCESS){
        cerr << "ERROR to create real addition of squared grid (zero) kernel " << err << endl;
        throw runtime_error("Failed to create real addition of squared grid (zero)  kernel.");
    }
    
    err = clGetKernelWorkGroupInfo(realAddSq0Kernel, *ctx_dev, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &_localRealAddSq0Size, NULL);
    if (err != CL_SUCCESS){
        cerr << "ERROR to inquire local work group size for real addition of squared grid (zero) kernel " << err << endl;
        throw runtime_error("Not possible to inquire local work group size of real addition of squared grid (zero) kernel.");
    }
    
    // SETUP THE POINTWISE COMPLEX MULTIPLICATION KERNEL
    // please note: both data and multipliers are actually packed complex
    this->cmplMultKernel = clCreateKernel(cartOCLOOCPProgram, "cmplMultGrid", &err);
    if (!cmplMultKernel || err != CL_SUCCESS){
        cerr << "ERROR to create complex multiplication kernel " << err << endl;
        throw runtime_error("Failed to create complex multiplication kernel.");
    }
    
    err = clGetKernelWorkGroupInfo(cmplMultKernel, *ctx_dev, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &_localCmplMultSize, NULL);
    if (err != CL_SUCCESS){
        cerr << "ERROR to inquire local work group size for complex multiplication kernel " << err << endl;
        throw runtime_error("Not possible to inquire local work group size of complex multiplication kernel.");
    }
    
    // SETUP THE POINTWISE COMPLEX * REAL MULTIPLICATION KERNEL
    // please note: data is actually packed complex
    this->cmplRealMultKernel = clCreateKernel(cartOCLOOCPProgram, "cmplRealMultGrid", &err);
    if (!cmplRealMultKernel || err != CL_SUCCESS){
        cerr << "ERROR to create complex/real multiplication kernel " << err << endl;
        throw runtime_error("Failed to create complex/real multiplication kernel.");
    }
    
    err = clGetKernelWorkGroupInfo(cmplRealMultKernel, *ctx_dev, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &_localCmplRealMultSize, NULL);
    if (err != CL_SUCCESS){
        cerr << "ERROR to inquire local work group size for complex times real multiplication kernel " << err << endl;
        throw runtime_error("Not possible to inquire local work group size of complex times real multiplication kernel.");
    }
    
    // SETUP THE PURE COMPLEX*COMPLEX MULTIPLICATION KERNEL
    this->pureCmplMultKernel = clCreateKernel(cartOCLOOCPProgram, "pureCmplMultGrid", &err);
    if (!cmplRealMultKernel || err != CL_SUCCESS){
        cerr << "ERROR to create pure complex multiplication kernel " << err << endl;
        throw runtime_error("Failed to create pure complex multiplication kernel.");
    }
    
    // SETUP THE POINTWISE GNORMS MULTIPLICATION KERNEL
    // please note: data is packed complex
    this->gNormMultKernel = clCreateKernel(cartOCLOOCPProgram, "gNormsMultGrid", &err);
    if (!gNormMultKernel || err != CL_SUCCESS){
        cerr << "ERROR to create gNorms multiplication kernel " << err << endl;
        throw runtime_error("Failed to create gNorms multiplication kernel.");
    }
    
    err = clGetKernelWorkGroupInfo(gNormMultKernel, *ctx_dev, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &_localGNormMultSize, NULL);
    if (err != CL_SUCCESS){
        cerr << "ERROR to inquire local work group size for gNorms multiplication kernel " << err << endl;
        throw runtime_error("Not possible to inquire local work group size of gNorms multiplication kernel.");
    }
    
    // SETUP THE TRIVIAL SUMMING KERNEL
    this->sumOverKernel = clCreateKernel(cartOCLOOCPProgram, "sumOverGrid", &err);
    if (!sumOverKernel || err != CL_SUCCESS){
        cerr << "ERROR to create sum-over kernel " << err << endl;
        throw runtime_error("Failed to create sum-over kernel.");
    }
    
    // SETUP THE ADDUP KERNEL
    this->addUpKernel = clCreateKernel(cartOCLOOCPProgram, "addUp", &err);
    if (!addUpKernel || err != CL_SUCCESS){
        cerr << "ERROR to create add-up kernel " << err << endl;
        throw runtime_error("Failed to create add-up kernel.");
    }
    
    err = clGetKernelWorkGroupInfo(sumOverKernel, *ctx_dev, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &_localSumOverSize, NULL);
    if (err != CL_SUCCESS){
        cerr << "ERROR to inquire local work group size for sum-over kernel " << err << endl;
        throw runtime_error("Not possible to inquire local work group size of sum-over kernel.");
    }
    
    // SETUP THE MIN/MAX TRIVIAL KERNEL
    this->minMaxGridKernel = clCreateKernel(cartOCLOOCPProgram, "minMaxGrid", &err);
    if (!minMaxGridKernel || err != CL_SUCCESS){
        cerr << "ERROR to create min/max fallback kernel " << err << endl;
        throw runtime_error("Failed to create min/max fallback kernel.");
    }
    
    // SETUP THE MIN/MAX FAST KERNEL
    this->minMaxFastKernel = clCreateKernel(cartOCLOOCPProgram, "minMaxFast", &err);
    if (!minMaxFastKernel || err != CL_SUCCESS){
        cerr << "ERROR to create min/max fast kernel " << err << endl;
        throw runtime_error("Failed to create min/max fast kernel.");
    }

    // SETUP THE REAL PRINTER
    this->realPrintKernel = clCreateKernel(cartOCLOOCPProgram, "realPrint", &err);
    if (!realPrintKernel || err != CL_SUCCESS){
        cerr << "ERROR to create real print kernel " << err << endl;
        throw runtime_error("Failed to create real print kernel.");
    }

    // SETUP THE COMPLEX PRINTER
    this->complexPrintKernel = clCreateKernel(cartOCLOOCPProgram, "complexPrint", &err);
    if (!complexPrintKernel || err != CL_SUCCESS){
        cerr << "ERROR to create complex print kernel " << err << endl;
        throw runtime_error("Failed to create complex print kernel.");
    }
    
    delete[] clLengths;
}

CartesianOCLOOPGrid::CartesianOCLOOPGrid(const CartesianOCLOOPGrid& orig)
: CartesianGrid(orig), _platforms(orig._platforms), _devices(orig._devices),
        _r2c_plan(orig._r2c_plan), _c2r_plan(orig._c2r_plan){
    
    this->_ctx = orig._ctx;
    clRetainContext(_ctx);

    this->_queue = orig._queue;
    clRetainCommandQueue(_queue);
    
    this->_noPlatforms = orig._noPlatforms;
    
    this->_noDevices = orig._noDevices;
    for(size_t x = 0; x < _noDevices; ++x){
        clRetainDevice(_devices.get()[x]);
    }

    this->cartOCLOOCPProgram = orig.cartOCLOOCPProgram;
    clRetainProgram(cartOCLOOCPProgram);
    
    this->realPrintKernel = orig.realPrintKernel;
    clRetainKernel(realPrintKernel);
    this->complexPrintKernel = orig.complexPrintKernel;
    clRetainKernel(complexPrintKernel);

    this->sqrtKernel = orig.sqrtKernel;
    clRetainKernel(sqrtKernel);
    this->_localSqrtSize = orig._localSqrtSize;
    
    this->powKernel = orig.powKernel;
    clRetainKernel(powKernel);
    this->_localPowSize = orig._localPowSize;
        
    this->addKernel = orig.addKernel;
    clRetainKernel(addKernel);

    this->fmaKernel = orig.fmaKernel;
    clRetainKernel(fmaKernel);

    this->realMultKernel = orig.realMultKernel;
    clRetainKernel(realMultKernel);
    this->_localRealMultSize = orig._localRealMultSize;
    
    this->mult2SqrtKernel = orig.mult2SqrtKernel;
    clRetainKernel(mult2SqrtKernel);
    
    this->realAddSq0Kernel = orig.realAddSq0Kernel;
    clRetainKernel(realAddSq0Kernel);
    this->_localRealAddSq0Size = orig._localRealAddSq0Size;
    
    this->realAddSqKernel = orig.realAddSqKernel;
    clRetainKernel(realAddSqKernel);
    this->_localRealAddSqSize = orig._localRealAddSqSize;
    
    this->cmplMultKernel = orig.cmplMultKernel;
    clRetainKernel(cmplMultKernel);
    this->_localCmplMultSize = orig._localCmplMultSize;

    this->cmplRealMultKernel = orig.cmplRealMultKernel;
    clRetainKernel(cmplRealMultKernel);
    this->_localCmplRealMultSize = orig._localCmplRealMultSize;
    
    this->pureCmplMultKernel = orig.pureCmplMultKernel;
    clRetainKernel(pureCmplMultKernel);
    
    this->gNormMultKernel = orig.gNormMultKernel;
    clRetainKernel(gNormMultKernel);
    this->_localGNormMultSize = orig._localGNormMultSize;
 
    this->sumOverKernel = orig.sumOverKernel;
    clRetainKernel(sumOverKernel);
    this->_localSumOverSize = orig._localSumOverSize;
    
    this->addUpKernel = orig.addUpKernel;
    clRetainKernel(addUpKernel);
    
    this->minMaxFastKernel = orig.minMaxFastKernel;
    clRetainKernel(minMaxFastKernel);
    
    this->minMaxGridKernel = orig.minMaxGridKernel;
    clRetainKernel(minMaxGridKernel);

    this->_realInSync = true;
    this->_reciInSync = true;
    this->_realReturned = true;
    this->_reciReturned = true;
    this->_realOnGPUInSync = false;
    this->_reciOnGPUInSync = false;
    
    // do lazy allocation of GPU memory to avoid having too many grids in GPU memory at the same time
    // this will obviously come at the expense of overhead but we can be smart when we need to compute
    // a more advanced quantity (say the divergence/gradient)
    this->_bufReal = NULL;
    this->_bufReci = NULL;
    
#ifndef LIBKEDF_LOWGPUMEMORY
    if(orig._gNormsBuf != NULL){
        // keep a reference to the gNorms
        this->_gNormsBuf = orig._gNormsBuf;
        clRetainMemObject(_gNormsBuf);
    } else {
        this->_gNormsBuf = NULL;
    }
#endif
        
    // allocate host memory buffer which can in turn be used by arma cube objects
    this->_bufferSizeHost  = _zDim*_yDim*2*(_xDim/2+1)*sizeof(double);
#ifdef LIBKEDF_NO_ALIGNED_MEMORY
    _hostMem = malloc(_bufferSizeHost);
#else
    // AVX wants at least 16, but rather 32 byte alignment for some features. This is basically 4 doubles in one line
    // upcoming AVX-512 operates on 64 (!) bytes of data simultaneously
    const size_t alignment = LIBKEDF_ALIGNMENT;
    const int error = posix_memalign(&_hostMem,alignment,_bufferSizeHost);
    if(error != 0){
        cerr << "ERROR: Can't get aligned host memory: " << error << ", size: " << _bufferSizeHost << ", alignment: " << alignment << endl;
        throw runtime_error("No aligned memory could be allocated.");
    }
#endif
    
    // use memory for arma backends. we define that the real storage is fully compressed (i.e., whatever extra bytes are in the end)
    this->_realGrid = new cube((double*)_hostMem,_xDim,_yDim,_zDim,false,true);
    this->_reciGrid = new cx_cube((complex<double>*)_hostMem,_xRecDim,_yRecDim,_zRecDim,false,true);
}
    
CartesianOCLOOPGrid::~CartesianOCLOOPGrid(){
    
    finalize(); // to make sure there is nothing left to do that involved buffers/kernels/queues we are about to discard

/*    err = clfftDestroyPlan(_r2c_plan);
    if(err != CL_SUCCESS){
        cerr << "ERROR in releasing R2C plan: " << err << endl;
    }

    err = clfftDestroyPlan(_c2r_plan);
    if(err != CL_SUCCESS){
        cerr << "ERROR in releasing C2R plan: " << err << endl;
    }*/
    
    for(size_t x = 0; x < _noDevices; ++x){
        clReleaseDevice(_devices.get()[x]);
    }
    
    free(_hostMem);
    freeGPUMemory(ALL);
    
    delete _realGrid;
    delete _reciGrid;
    
#ifndef LIBKEDF_LOWGPUMEMORY
    if(_gNormsBuf != NULL){
        clReleaseMemObject(_gNormsBuf);
    }
#endif
    
    clReleaseContext(_ctx);
    clReleaseCommandQueue(_queue);
    
    clReleaseKernel(realPrintKernel);
    clReleaseKernel(complexPrintKernel);
    clReleaseKernel(sqrtKernel);
    clReleaseKernel(powKernel);
    clReleaseKernel(realMultKernel);
    clReleaseKernel(mult2SqrtKernel);
    clReleaseKernel(realAddSq0Kernel);
    clReleaseKernel(realAddSqKernel);
    clReleaseKernel(cmplMultKernel);
    clReleaseKernel(pureCmplMultKernel);
    clReleaseKernel(gNormMultKernel);
    clReleaseKernel(sumOverKernel);
    clReleaseKernel(addUpKernel);
    clReleaseKernel(addKernel);
    clReleaseKernel(fmaKernel);
    clReleaseKernel(minMaxGridKernel);
    clReleaseKernel(minMaxFastKernel);
    
    clReleaseProgram(cartOCLOOCPProgram);
}
    
unique_ptr<FourierGrid> CartesianOCLOOPGrid::createFourierDuplicate() const {
    
    unique_ptr<FourierGrid> dup = this->duplicate();
    
    return dup;
}
    
unique_ptr<CartesianOCLOOPGrid> CartesianOCLOOPGrid::duplicate() const {

#ifdef LIBKEDF_DEBUG
        cout << "DEBUG: creating Fourier duplicate of OCL OOP grid." << endl;
        cout << "DEBUG: sync status: " << _realInSync << "\t" << _reciInSync << "\t" << _realOnGPUInSync << "\t" << _reciOnGPUInSync << endl;
#endif

    unique_ptr<CartesianOCLOOPGrid> dup(new CartesianOCLOOPGrid(*this));
    
    // only copy the state if it is in sync (i.e., if it is worth having) *and* we are not about to copy the GPU buffer
    // rational: if we have and maintain GPU state, that's the more valuable for us to have and the CPU state is
    // probably going to be out of sync anyways with the next computation
    if((_realInSync || _reciInSync) && !_reciOnGPUInSync && !_realOnGPUInSync){
#ifdef LIBKEDF_DEBUG
        cout << "DEBUG: copying current OCL OOP host state." << endl;
#endif
        memcpy(dup->_hostMem,_hostMem,this->_bufferSizeHost);
        dup->_realInSync = _realInSync;
        dup->_reciInSync = _reciInSync;
    } else {
        dup->_realInSync = false;
        dup->_reciInSync = false;
    }
    
    if(_reciOnGPUInSync){
        
#ifdef LIBKEDF_DEBUG
        cout << "DEBUG: copying current OCL OOP reci GPU state." << endl;
#endif
        
        // allocate GPU memory for this
        dup->allocateGPUMemory(RECI);
        cl_int err = clEnqueueCopyBuffer (_queue, _bufReci, dup->_bufReci, 0, 0, _bufferSizeHost, 0, NULL, NULL);
        if(err != CL_SUCCESS){
            cerr << "ERROR in copying reci buffer " << err << endl;
            throw runtime_error("Failed to reci buffer.");
        }
    }
    
    if(_realOnGPUInSync){
        
#ifdef LIBKEDF_DEBUG
        cout << "DEBUG: copying current OCL OOP real GPU state." << endl;
#endif
        
        // allocate GPU memory for this
        dup->allocateGPUMemory(REAL);
        cl_int err = clEnqueueCopyBuffer (_queue, _bufReal, dup->_bufReal, 0, 0, _bufferSizeHost, 0, NULL, NULL);
        if(err != CL_SUCCESS){
            cerr << "ERROR in copying real buffer " << err << endl;
            throw runtime_error("Failed to real buffer.");
        }
    }
    
    dup->_realReturned = true;
    dup->_reciReturned = true;
    dup->_realOnGPUInSync = _realOnGPUInSync;
    dup->_reciOnGPUInSync = _reciOnGPUInSync;

    return dup;
}

unique_ptr<FourierGrid> CartesianOCLOOPGrid::createFourierEmptyDuplicate() const {
    
    unique_ptr<FourierGrid> dup = this->emptyDuplicate();
    
    return dup;
}

unique_ptr<CartesianOCLOOPGrid> CartesianOCLOOPGrid::emptyDuplicate() const {
    
#ifdef LIBKEDF_DEBUG
        cout << "DEBUG: creating empty Fourier duplicate of OCL OOP grid." << endl;
#endif

    unique_ptr<CartesianOCLOOPGrid> dup(new CartesianOCLOOPGrid(*this));
    
    dup->_realInSync = true;
    dup->_reciInSync = true;
    dup->_realReturned = true;
    dup->_reciReturned = true;
    dup->_realOnGPUInSync = false;
    dup->_reciOnGPUInSync = false;
    
    return dup;
}
    
cube* CartesianOCLOOPGrid::getRealGrid(){
    
    if(!_reciReturned){
        throw runtime_error("reciprocal grid must be returned before getting real grid");
    }
    if(!_realReturned){
        throw runtime_error("real grid must be returned before getting real grid");
    }
    
    if(!_realInSync){
        
        if(!_realOnGPUInSync){
            // do FFT
            enqueueBackwardTransform();
        }
        
        transferRealFromGPU();
    }
    
    _realReturned = false;
    _realOnGPUInSync = false;
    _reciOnGPUInSync = false;
    _reciInSync = false;
    _realInSync = true;
    
    return _realGrid;
}

cx_cube* CartesianOCLOOPGrid::getReciprocalGrid(){
    
    if(!_reciReturned){
        throw runtime_error("reciprocal grid must be returned before getting reciprocal grid");
    }
    if(!_realReturned){
        throw runtime_error("real grid must be returned before getting reciprocal grid");
    }
    
    if(!_reciInSync){

        if(!_reciOnGPUInSync){
            // do FFT
            enqueueForwardTransform();
        }
        
        transferReciFromGPU();
    }
    
    _reciReturned = false;
    _realOnGPUInSync = false;
    _reciOnGPUInSync = false;
    _realInSync = false;
    _reciInSync = true;
    
    return _reciGrid;
}
    
void CartesianOCLOOPGrid::complete(cube* realGrid){
    
    if(realGrid == NULL){
        throw runtime_error("real grid null");
    }
    
    // compare addresses
    if(realGrid != _realGrid){
        // problem
        throw runtime_error("real grid pointer returned not equal to real grid pointer given.");
    }
    
    _realReturned = true;
    
    realGrid = NULL;
}

void CartesianOCLOOPGrid::completeReciprocal(cx_cube* reciprocalGrid){
    
    if(reciprocalGrid == NULL){
        throw runtime_error("reciprocal grid null");
    }
    
    // compare addresses
    if(reciprocalGrid != _reciGrid){
        // problem
        throw runtime_error("reciprocal grid pointer returned not equal to reciprocal grid pointer given.");
    }
    
    _reciReturned = true;
    
    reciprocalGrid = NULL;
}

void CartesianOCLOOPGrid::resetToReal(){
    
    if(!_reciReturned || !_realReturned){
        throw runtime_error("must return all grid pointers before reset.");
    }
    
    _reciInSync = false;
    _realInSync = true;
    _realOnGPUInSync = false;
    _reciOnGPUInSync = false;
}

void CartesianOCLOOPGrid::resetToReciprocal(){
    
    if(!_reciReturned || !_realReturned){
        throw runtime_error("must return all grid pointers before reset.");
    }
    
    _reciInSync = true;
    _realInSync = false;
    _realOnGPUInSync = false;
    _reciOnGPUInSync = false;
}
    
const cube* CartesianOCLOOPGrid::readRealGrid() {
    
    bool origRealOnGPu = _realOnGPUInSync;
    
    getRealGrid();
    _realReturned = true; // XXX: is this always true?
    _realOnGPUInSync = origRealOnGPu;
    
    return _realGrid;
}

const cube* CartesianOCLOOPGrid::tryReadRealGrid() const {
    
    if(!_reciReturned || !_realReturned || !_realInSync){
        return NULL;
    }
        
    return _realGrid;
}

const cx_cube* CartesianOCLOOPGrid::readReciprocalGrid() {
    
    bool origReciOnGPu = _reciOnGPUInSync;
    
    getReciprocalGrid();
    _reciReturned = true; // XXX: is this always true?
    _reciOnGPUInSync = origReciOnGPu;
    
    return _reciGrid;
}

void CartesianOCLOOPGrid::allocateGPUMemory(const gpumem_allocation_t alloc){
    
    if(alloc == ALL || alloc == REAL){
        if(_bufReal == NULL){
        
#ifdef LIBKEDF_DEBUG
            // XXX this is a bit too much, but who cares for those few elements
            cout << "DEBUG: Allocating real " << _bufferSizeHost << " bytes in context " << _ctx << endl;
#endif
        
            cl_int err;
            _bufReal = clCreateBuffer(_ctx, CL_MEM_READ_WRITE, _bufferSizeHost, NULL, &err);
            if(err != CL_SUCCESS){
                cerr << "ERROR in allocating real GPU memory of size " << _bufferSizeHost << " with error " << err << endl;
                throw runtime_error("Failed to allocate GPU memory");
            }
        }
    }
    
    if(alloc == ALL || alloc == RECI){
    
        if(_bufReci == NULL){
        
#ifdef LIBKEDF_DEBUG
            cout << "DEBUG: Allocating reci " << _bufferSizeHost << " bytes in context " << _ctx << endl;
#endif
            
            cl_int err;
            _bufReci = clCreateBuffer(_ctx, CL_MEM_READ_WRITE, _bufferSizeHost, NULL, &err);
            if(err != CL_SUCCESS){
                cerr << "ERROR in allocating reci GPU memory of size " << _bufferSizeHost << " with error " << err << endl;
                throw runtime_error("Failed to allocate GPU memory");
            }
        }
    }
}

void CartesianOCLOOPGrid::freeGPUMemory(const gpumem_allocation_t alloc) {
    
    if(alloc == ALL || alloc == RECI){
        if(_bufReal != NULL){

#ifdef LIBKEDF_DEBUG
            cout << "DEBUG: Freeing real " << _bufferSizeHost << " bytes." << endl;
#endif

            cl_int err = clReleaseMemObject(_bufReal);
            if(err != CL_SUCCESS){
                cerr << "ERROR in freeing real GPU memory: " << err << endl;
            }
        }
    }
    
    if(alloc == ALL || alloc == RECI){
        if(_bufReci != NULL){

#ifdef LIBKEDF_DEBUG
            cout << "DEBUG: Freeing reci " << _bufferSizeHost << " bytes." << endl;
#endif

            cl_int err = clReleaseMemObject(_bufReci);
            if(err != CL_SUCCESS){
                cerr << "ERROR in freeing reci GPU memory: " << err << endl;
            }
        }
    }
}

unique_ptr<CartesianOCLOOPGrid> CartesianOCLOOPGrid::laplacian() {
    return GridComputer::laplacian<CartesianOCLOOPGrid>(this);
}

void CartesianOCLOOPGrid::transferGNorms(){
#ifndef LIBKEDF_LOWGPUMEMORY
    if(_gNormsBuf == NULL){
        // we need to allocate and transfer
        
        cl_int err;
        const size_t reciPoints = _xRecDim*_yRecDim*_zRecDim;
        const size_t normsSize = sizeof(double)*reciPoints;
        
#ifdef LIBKEDF_DEBUG
        cout << "DEBUG: Allocating " << normsSize << " bytes for gNorms in context " << _ctx << endl;
#endif
        
        this->_gNormsBuf = clCreateBuffer(_ctx, CL_MEM_READ_ONLY, normsSize, NULL, &err);
        if(err != CL_SUCCESS){
            cerr << "ERROR in allocating GPU memory of size " << normsSize << " with error " << err << endl;
            throw runtime_error("Failed to allocate GPU memory");
        }
        
        const cube* gNorms = this->getGNorms();
        err = clEnqueueWriteBuffer(_queue, _gNormsBuf, CL_FALSE, 0, normsSize, gNorms->memptr(), 0, NULL, NULL );
        if(err != CL_SUCCESS){
            cerr << "ERROR to write gnorms data to GPU " << err << endl;
            throw runtime_error("Failed to write gnorms data to GPU.");
        }
    }
#endif
}


void CartesianOCLOOPGrid::multiplyGNorms(){

#ifdef LIBKEDF_DEBUG
    cout << "DEBUG: multiplyGNorms called in OCL grid." << endl;
#endif
    
    if(!_reciOnGPUInSync){        
        if(!_reciInSync){
            enqueueForwardTransform();
        } else {
            transferReciToGPU();
        }
    }
    
    // allocate space for the gnorms on the GPU and transfer them
    
    cl_int err;
    const size_t reciPoints = _xRecDim*_yRecDim*_zRecDim;
    const size_t normsSize = sizeof(double)*reciPoints;
    
#ifndef LIBKEDF_LOWGPUMEMORY
    if(_gNormsBuf == NULL){
        // we need to allocate and transfer
        
#ifdef LIBKEDF_DEBUG
        cout << "DEBUG: Allocating " << normsSize << " bytes for gNorms in context " << _ctx << endl;
#endif
        
        this->_gNormsBuf = clCreateBuffer(_ctx, CL_MEM_READ_ONLY, normsSize, NULL, &err);
        if(err != CL_SUCCESS){
            cerr << "ERROR in allocating GPU memory of size " << normsSize << " with error " << err << endl;
            throw runtime_error("Failed to allocate GPU memory");
        }
        
        const cube* gNorms = this->getGNorms();
        err = clEnqueueWriteBuffer(_queue, _gNormsBuf, CL_FALSE, 0, normsSize, gNorms->memptr(), 0, NULL, NULL );
        if(err != CL_SUCCESS){
            cerr << "ERROR to write gnorms data to GPU " << err << endl;
            throw runtime_error("Failed to write gnorms data to GPU.");
        }
    }
    
    // now simply run the multiplication
    err  = clSetKernelArg(gNormMultKernel, 0, sizeof(cl_mem), &_bufReci);
    err |= clSetKernelArg(gNormMultKernel, 1, sizeof(cl_mem), &_gNormsBuf);
    if(err != CL_SUCCESS){
        cerr << "ERROR in setting a gNorms multiplication kernel argument: " << err << endl;
        throw runtime_error("Failed to set a gNorms multiplication kernel argument.");
    }
#else
    
#ifdef LIBKEDF_DEBUG
    cout << "DEBUG: Allocating " << normsSize << " bytes for gNorms in context " << _ctx << endl;
#endif
    
    cl_mem bufNorms = clCreateBuffer(_ctx, CL_MEM_READ_ONLY, normsSize, NULL, &err);
    if(err != CL_SUCCESS){
        cerr << "ERROR in allocating GPU memory of size " << normsSize << " with error " << err << endl;
        throw runtime_error("Failed to allocate GPU memory");
    }
    const cube* gNorms = this->getGNorms();
    err = clEnqueueWriteBuffer(_queue, bufNorms, CL_FALSE, 0, normsSize, gNorms->memptr(), 0, NULL, NULL );
    if(err != CL_SUCCESS){
        cerr << "ERROR to write gnorms data to GPU " << err << endl;
        throw runtime_error("Failed to write gnorms data to GPU.");
    }

    // now run the multiplication actually
    
    err  = clSetKernelArg(gNormMultKernel, 0, sizeof(cl_mem), &_bufReci);
    err |= clSetKernelArg(gNormMultKernel, 1, sizeof(cl_mem), &bufNorms);
    if(err != CL_SUCCESS){
        cerr << "ERROR in setting a gNorms multiplication kernel argument: " << err << endl;
        throw runtime_error("Failed to set a gNorms multiplication kernel argument.");
    }
    
#endif
 
    // enqueue the kernel over the entire "1D'd" reciprocal grid
    err = clEnqueueNDRangeKernel(_queue, gNormMultKernel, 1, NULL, &reciPoints, NULL, 0, NULL, NULL);
    if (err != CL_SUCCESS){
        cerr << "ERROR in enqueueing gNorms multiplication kernel failed. " << err << endl;
        throw runtime_error("1D-enqueue of gNorms multiplication kernel failed.");
    }
    
    _reciOnGPUInSync = true;
    _realOnGPUInSync = false;
    _realInSync = false;
    _reciInSync = false; // not transferred yet
    

#ifdef LIBKEDF_LOWGPUMEMORY    
    err = clReleaseMemObject(bufNorms);
    if(err != CL_SUCCESS){
        cerr << "ERROR in freeing GPU memory: " << err << endl;
    }
#endif
    
#ifdef LIBKEDF_DEBUG
    cout << "DEBUG: multipyGNorms finished in OCL grid." << endl;
#endif
}

void CartesianOCLOOPGrid::addGrid(CartesianOCLOOPGrid *other){
    
#ifdef LIBKEDF_DEBUG
    cout << "DEBUG: addGrid in CartesianOCLOOPGrid called." << endl;
#endif
    
    if(!_realOnGPUInSync){
        if(!_realInSync){
            enqueueBackwardTransform();
        } else {
            transferRealToGPU();
        }
    }
    
    if(!other->_realOnGPUInSync){
        if(!other->_realInSync){
            other->enqueueBackwardTransform();
        } else {
            other->transferRealToGPU();
        }
    }
    
    // now both real grids are on the GPU: add them
    const size_t realPoints = _xDim*_yDim*_zDim;
    cl_int err;
    err  = clSetKernelArg(addKernel, 0, sizeof(cl_mem), &_bufReal);
    err |= clSetKernelArg(addKernel, 1, sizeof(cl_mem), &other->_bufReal);
    if(err != CL_SUCCESS){
        cerr << "ERROR in setting a add kernel argument: " << err << endl;
        throw runtime_error("Failed to set a add kernel argument.");
    }
        
    // enqueue the kernel over the entire "1D'd" grid
    const size_t points = realPoints/getVectortypeAlignment();
    err = clEnqueueNDRangeKernel(_queue, addKernel, 1, NULL, &points, NULL, 0, NULL, NULL);
    if (err != CL_SUCCESS){
        throw runtime_error("1D-enqueue of add kernel failed.");
    }

    _reciOnGPUInSync = false;
    _realOnGPUInSync = true;
    _realInSync = false; // not transferred yet
    _reciInSync = false;
}

void CartesianOCLOOPGrid::fmaGrid(const double mult, CartesianOCLOOPGrid *other){
    
#ifdef LIBKEDF_DEBUG
    cout << "DEBUG: fma in CartesianOCLOOPGrid called." << endl;
#endif
    
    if(abs(mult) < Grid::NUMERICALACCURACY){
        return;
    }
    if(abs(1.0-mult) < Grid::NUMERICALACCURACY){
        addGrid(other);
        return;
    }
    
    if(!_realOnGPUInSync){
        if(!_realInSync){
            enqueueBackwardTransform();
        } else {
            transferRealToGPU();
        }
    }
    
    if(!other->_realOnGPUInSync){
        if(!other->_realInSync){
            other->enqueueBackwardTransform();
        } else {
            other->transferRealToGPU();
        }
    }
    
    // now both real grids are on the GPU: fma them
    const size_t realPoints = _xDim*_yDim*_zDim;
    cl_int err;
    err  = clSetKernelArg(fmaKernel, 0, sizeof(cl_mem), &_bufReal);
    err |= clSetKernelArg(fmaKernel, 1, sizeof(cl_mem), &other->_bufReal);
    err |= clSetKernelArg(fmaKernel, 2, sizeof(double), &mult);
    if(err != CL_SUCCESS){
        cerr << "ERROR in setting a fma kernel argument: " << err << endl;
        throw runtime_error("Failed to set a fma kernel argument.");
    }
        
    // enqueue the kernel over the entire "1D'd" grid
    err = clEnqueueNDRangeKernel(_queue, fmaKernel, 1, NULL, &realPoints, NULL, 0, NULL, NULL);
    if (err != CL_SUCCESS){
        throw runtime_error("1D-enqueue of fma kernel failed.");
    }

    _reciOnGPUInSync = false;
    _realOnGPUInSync = true;
    _realInSync = false; // not transferred yet
    _reciInSync = false;
}

void CartesianOCLOOPGrid::multiplyElementwise(CartesianOCLOOPGrid* grid){
    
#ifdef LIBKEDF_DEBUG
    cout << "DEBUG: calling multiplyElementwise in CartesianOCLOOPGrid." << endl;
#endif
    
    if(!_realOnGPUInSync){
        if(!_realInSync){
            enqueueBackwardTransform();
        } else {
            transferRealToGPU();
        }
    }
    
    if(!grid->_realOnGPUInSync){
        if(!grid->_realInSync){
            grid->enqueueBackwardTransform();
        } else {
            grid->transferRealToGPU();
        }
    }
    
    // now both real grids are on the GPU: multiply them
    const size_t realPoints = _xDim*_yDim*_zDim;
    cl_int err;
    err  = clSetKernelArg(realMultKernel, 0, sizeof(cl_mem), &_bufReal);
    err |= clSetKernelArg(realMultKernel, 1, sizeof(cl_mem), &grid->_bufReal);
    if(err != CL_SUCCESS){
        cerr << "ERROR in setting a elementwise multiplication kernel argument: " << err << endl;
        throw runtime_error("Failed to set a elementwise multiplication kernel argument.");
    }
        
    // enqueue the kernel over the entire "1D'd" grid
    const size_t points = realPoints/getVectortypeAlignment();
    err = clEnqueueNDRangeKernel(_queue, realMultKernel, 1, NULL, &points, NULL, 0, NULL, NULL);
    if (err != CL_SUCCESS){
        throw runtime_error("1D-enqueue of elementwise multiplication kernel failed.");
    }

    _reciOnGPUInSync = false;
    _realOnGPUInSync = true;
    _realInSync = false; // not transferred yet
    _reciInSync = false;
    
}

void CartesianOCLOOPGrid::multiplyTwoSqrtOf(CartesianOCLOOPGrid* other){

#ifdef LIBKEDF_DEBUG
    cout << "DEBUG: calling multiplyTwoSqrtOf in CartesianOCLOOPGrid." << endl;
#endif
    
    if(!_realOnGPUInSync){
        if(!_realInSync){
            enqueueBackwardTransform();
        } else {
            transferRealToGPU();
        }
    }
    
    if(!other->_realOnGPUInSync){
        if(!other->_realInSync){
            other->enqueueBackwardTransform();
        } else {
            other->transferRealToGPU();
        }
    }

    // now both real grids are on the GPU: multiply them
    const size_t realPoints = _xDim*_yDim*_zDim;
    cl_int err;
    err  = clSetKernelArg(mult2SqrtKernel, 0, sizeof(cl_mem), &_bufReal);
    err |= clSetKernelArg(mult2SqrtKernel, 1, sizeof(cl_mem), &other->_bufReal);
    if(err != CL_SUCCESS){
        cerr << "ERROR in setting a elementwise multiplication with two sqrt kernel argument: " << err << endl;
        throw runtime_error("Failed to set a elementwise multiplication with two sqrt kernel argument.");
    }
        
    // enqueue the kernel over the entire "1D'd" grid
    const size_t points = realPoints/getVectortypeAlignment();
    err = clEnqueueNDRangeKernel(_queue, mult2SqrtKernel, 1, NULL, &points, NULL, 0, NULL, NULL);
    if (err != CL_SUCCESS){
        throw runtime_error("1D-enqueue of elementwise multiplication with two sqrt kernel failed.");
    }

    _reciOnGPUInSync = false;
    _realOnGPUInSync = true;
    _realInSync = false; // not transferred yet
    _reciInSync = false;
}

void CartesianOCLOOPGrid::multiplyGVectorsX(){
    
    const cube* vectors = this->getGVectorsX();
    this->multiplyGVectors(vectors);
}

void CartesianOCLOOPGrid::multiplyGVectorsY(){
    
    const cube* vectors = this->getGVectorsY();
    this->multiplyGVectors(vectors);
}

void CartesianOCLOOPGrid::multiplyGVectorsZ(){
    
    const cube* vectors = this->getGVectorsZ();
    this->multiplyGVectors(vectors);
}

void CartesianOCLOOPGrid::multiplyGVectors(const cube* vectors){

    if(!_reciOnGPUInSync){
        if(!_reciInSync){
            enqueueForwardTransform();
        } else {
            transferReciToGPU();
        }
    }

    // transfer the gVectors to the GPU
    cl_int err;
    const size_t reciPoints = _xRecDim*_yRecDim*_zRecDim;
    const size_t vecsSize = sizeof(double)*reciPoints;
    
#ifdef LIBKEDF_DEBUG
    cout << "DEBUG: Allocating " << vecsSize << " bytes for gVectors in context " << _ctx << endl;
#endif
    
    cl_mem bufVecs = clCreateBuffer(_ctx, CL_MEM_READ_ONLY, vecsSize, NULL, &err);
    if(err != CL_SUCCESS){
       cerr << "ERROR in allocating GPU memory of size " << vecsSize << " with error " << err << endl;
        throw runtime_error("Failed to allocate GPU memory");
    }

    err = clEnqueueWriteBuffer(_queue, bufVecs, CL_FALSE, 0, vecsSize, vectors->memptr(), 0, NULL, NULL );
    if(err != CL_SUCCESS){
        cerr << "ERROR to write g vectors data to GPU " << err << endl;
        throw runtime_error("Failed to write g vectors data to GPU.");
    }

    // now run the multiplication actually

    err  = clSetKernelArg(pureCmplMultKernel, 0, sizeof(cl_mem), &_bufReci);
    err |= clSetKernelArg(pureCmplMultKernel, 1, sizeof(cl_mem), &bufVecs);
    if(err != CL_SUCCESS){
        cerr << "ERROR in setting a g vectors multiplication kernel argument: " << err << endl;
        throw runtime_error("Failed to set a g vectors multiplication kernel argument.");
    }
        
    // enqueue the kernel over the entire "1D'd" grid
    err = clEnqueueNDRangeKernel(_queue, pureCmplMultKernel, 1, NULL, &reciPoints, NULL, 0, NULL, NULL);
    if (err != CL_SUCCESS){
        throw runtime_error("1D-enqueue of g vectors multiplication kernel failed.");
    }

    _reciOnGPUInSync = true;
    _realOnGPUInSync = false;
    _realInSync = false;
    _reciInSync = false; // not transferred yet
    
    
    err = clReleaseMemObject(bufVecs);
    if(err != CL_SUCCESS){
        cerr << "ERROR in freeing GPU memory: " << err << endl;
    }
    
#ifdef LIBKEDF_DEBUG
    cout << "DEBUG: multiplyGVectors finished in OCL grid." << endl;
#endif
}

unique_ptr<CartesianOCLOOPGrid> CartesianOCLOOPGrid::gradientSquared(){
    
#ifdef LIBKEDF_DEBUG
    cout << "DEBUG: doing gradient squared calculation in OCL OOP grid." << endl;
#endif
        
    unique_ptr<CartesianOCLOOPGrid> gradientSquared = this->emptyDuplicate();
    gradientSquared->resetToReal();
    gradientSquared->markRealDirty(); // because we haven't transferred it yet
    gradientSquared->markReciDirty();
    gradientSquared->markReciGPUDirty();
    gradientSquared->markRealGPUClean(); // by definition
    cl_mem realBuff = gradientSquared->getRealGPUBuffer();
        
    const size_t nSlices = this->getGridPointsZ();
    const size_t nRows = this->getGridPointsY();
    const size_t nCols = this->getGridPointsX();
    const size_t nReciSlices = this->getReciGridPointsZ();
    const size_t nReciRows = this->getReciGridPointsY();
    const size_t nReciCols = this->getReciGridPointsX();
    const size_t total = nSlices*nRows*nCols;
    const size_t vecSize = this->getVectortypeAlignment();
    const size_t totEnqSize = total/vecSize;
    const size_t totalReci =  nReciSlices*nReciRows*nReciCols;
    const size_t totalReciMem = totalReci*2*sizeof(double);

    // we will need this for each directional term
    unique_ptr<CartesianOCLOOPGrid> recGrid = this->duplicate();
    if(!recGrid->_reciOnGPUInSync){
        if(!recGrid->_reciInSync){
            recGrid->enqueueForwardTransform();
        } else {
            recGrid->transferReciToGPU();
        }
    }
        
    // X-contribution
    unique_ptr<CartesianOCLOOPGrid> workGrid = recGrid->duplicate();
    workGrid->multiplyGVectorsX(); // now contains g-Vectors-X*FFT(density)
    workGrid->enqueueBackwardTransform();
    
    cl_mem workBuff = workGrid->getRealGPUBuffer();
    
    cl_int err;
    err  = clSetKernelArg(realAddSq0Kernel, 0, sizeof(cl_mem), &realBuff);
    err |= clSetKernelArg(realAddSq0Kernel, 1, sizeof(cl_mem), &workBuff);
    if(err != CL_SUCCESS){
        cerr << "ERROR in setting a real addition of squared grid kernel (zero) argument: " << err << endl;
        throw runtime_error("Failed to set a real addition of squared grid (zero) argument.");
    }
        
    // enqueue the kernel over the entire "1D'd" grid
    err = clEnqueueNDRangeKernel(_queue, realAddSq0Kernel, 1, NULL, &totEnqSize, NULL, 0, NULL, NULL);
    if (err != CL_SUCCESS){
        throw runtime_error("1D-enqueue of real addition of squared grid kernel (zero) failed.");
    }
        
    // Y-contribution
    
    cl_mem workReciBuff = workGrid->getReciGPUBuffer();
    cl_mem refReciBuff = recGrid->getReciGPUBuffer();
    // copy over and fix state
    err = clEnqueueCopyBuffer(_queue, refReciBuff, workReciBuff, 0, 0, totalReciMem, 0, NULL, NULL);
    if (err != CL_SUCCESS){
        throw runtime_error("Enqueue of reciprocal buffer copy failed.");
    }
    workGrid->markReciDirty();
    workGrid->markRealDirty();
    workGrid->markRealGPUDirty();
    workGrid->markReciGPUClean();
    
    workGrid->multiplyGVectorsY(); // now contains g-Vectors-Y*FFT(density)
    workGrid->enqueueBackwardTransform();
    
    err  = clSetKernelArg(realAddSqKernel, 0, sizeof(cl_mem), &realBuff);
    err |= clSetKernelArg(realAddSqKernel, 1, sizeof(cl_mem), &workBuff);
    if(err != CL_SUCCESS){
        cerr << "ERROR in setting a real addition of squared grid kernel argument: " << err << endl;
        throw runtime_error("Failed to set a real addition of squared grid argument.");
    }
        
    // enqueue the kernel over the entire "1D'd" grid
    err = clEnqueueNDRangeKernel(_queue, realAddSqKernel, 1, NULL, &totEnqSize, NULL, 0, NULL, NULL);
    if (err != CL_SUCCESS){
        throw runtime_error("1D-enqueue of real addition of squared grid kernel failed.");
    }
                
    // Z-contribution
    // copy over and fix state
    err = clEnqueueCopyBuffer(_queue, refReciBuff, workReciBuff, 0, 0, totalReciMem, 0, NULL, NULL);
    if (err != CL_SUCCESS){
        throw runtime_error("Enqueue of reciprocal buffer copy failed.");
    }
    workGrid->markReciDirty();
    workGrid->markRealDirty();
    workGrid->markRealGPUDirty();
    workGrid->markReciGPUClean();
    
    workGrid->multiplyGVectorsZ(); // now contains g-Vectors-Z*FFT(density)
    workGrid->enqueueBackwardTransform();
    
    err  = clSetKernelArg(realAddSqKernel, 0, sizeof(cl_mem), &realBuff);
    err |= clSetKernelArg(realAddSqKernel, 1, sizeof(cl_mem), &workBuff);
    if(err != CL_SUCCESS){
        cerr << "ERROR in setting a real addition of squared grid kernel argument: " << err << endl;
        throw runtime_error("Failed to set a real addition of squared grid argument.");
    }
        
    // enqueue the kernel over the entire "1D'd" grid
    err = clEnqueueNDRangeKernel(_queue, realAddSqKernel, 1, NULL, &totEnqSize, NULL, 0, NULL, NULL);
    if (err != CL_SUCCESS){
        throw runtime_error("1D-enqueue of real addition of squared grid kernel failed.");
    }
    
#ifdef LIBKEDF_DEBUG
    cout << "DEBUG: done with gradient squared calculation in OCL OOP grid." << endl;
#endif
    
    return gradientSquared;
}

void CartesianOCLOOPGrid::sqrtGrid(){
    
#ifdef LIBKEDF_DEBUG
    cout << "DEBUG: sqrtGrid called in OCL grid." << endl;
#endif

    if(!_realOnGPUInSync){        
        if(!_realInSync){
            enqueueBackwardTransform();
        } else {
            transferRealToGPU();
        }
    }

    cl_int err = clSetKernelArg(sqrtKernel, 0, sizeof(cl_mem), &_bufReal);
    if(err != CL_SUCCESS){
        cerr << "ERROR in setting sqrt kernel argument: " << err << endl;
        throw runtime_error("Failed to set sqrt kernel argument.");
    }
        
    // enqueue the kernel over the entire "1D'd" grid
    const size_t points = _noGridPoints/getVectortypeAlignment();
    err = clEnqueueNDRangeKernel(_queue, sqrtKernel, 1, NULL, &points, NULL, 0, NULL, NULL);
    if (err != CL_SUCCESS){
        cerr << "ERROR in enqueueing sqrt kernel: " << err << endl;
        throw runtime_error("1D-enqueue of sqrt kernel failed.");
    }
    
    _reciOnGPUInSync = false;
    _realOnGPUInSync = true;
    _realInSync = false; // not transferred yet
    _reciInSync = false;
    
    
#ifdef LIBKEDF_DEBUG
    cout << "DEBUG: sqrtGrid finished in OCL grid." << endl;
#endif
    
}

void CartesianOCLOOPGrid::powGrid(const double exponent) {
    
#ifdef LIBKEDF_DEBUG
    cout << "DEBUG: powGrid called in OCL grid." << endl;
#endif
    
    if(!_realOnGPUInSync){        
        if(!_realInSync){
            enqueueBackwardTransform();
        } else {
            transferRealToGPU();
        }
    }

    cl_int err;
    err  = clSetKernelArg(powKernel, 0, sizeof(cl_mem), &_bufReal);
    err |= clSetKernelArg(powKernel, 1, sizeof(double), &exponent);
    if(err != CL_SUCCESS){
        cerr << "ERROR in setting a pow kernel argument: " << err << endl;
        throw runtime_error("Failed to set a pow kernel argument.");
    }
        
    // enqueue the kernel over the entire "1D'd" grid
    const size_t points = _noGridPoints/getVectortypeAlignment();
    err = clEnqueueNDRangeKernel(_queue, powKernel, 1, NULL, &points, NULL, 0, NULL, NULL);
    if (err != CL_SUCCESS){
        throw runtime_error("1D-enqueue of pow kernel failed.");
    }
    
    _reciOnGPUInSync = false;
    _realOnGPUInSync = true;
    _realInSync = false; // not transferred yet
    _reciInSync = false;
    
    
#ifdef LIBKEDF_DEBUG
    cout << "DEBUG: powGrid finished in OCL grid." << endl;
#endif    
}

double CartesianOCLOOPGrid::sumOver(){
    
#ifdef LIBKEDF_DEBUG
    cout << "DEBUG: Calling sum-over kernel in OCL grid." << endl;
#endif
    
    if(!_realOnGPUInSync){        
        if(!_realInSync){
            enqueueBackwardTransform();
        } else {
            transferRealToGPU();
        }
    }
    
    
#ifdef LIBKEDF_SLOW_INTEGRATION
    
    const size_t rows = this->_xDim;
    const size_t cols = this->_yDim;
    const size_t slices = this->_zDim;
    const cl_int sliceLength = rows*cols;
    const size_t memSize = slices*sizeof(double);
    
    double tmpHost[slices];
    
    // allocate tmp scratch
    cl_int err;
    cl_mem tmp = clCreateBuffer(_ctx, CL_MEM_WRITE_ONLY, memSize, NULL, &err);
    if(err != CL_SUCCESS){
        cerr << "ERROR in allocating GPU memory of size " << memSize << " with error " << err << endl;
        throw runtime_error("Failed to allocate GPU memory");
    }
    
    err  = clSetKernelArg(sumOverKernel, 0, sizeof(cl_mem), &_bufReal);
    err |= clSetKernelArg(sumOverKernel, 1, sizeof(cl_mem), &tmp);
    err |= clSetKernelArg(sumOverKernel, 2, sizeof(int), &sliceLength);
    if(err != CL_SUCCESS){
        cerr << "ERROR in setting a sum-over kernel argument: " << err << endl;
        throw runtime_error("Failed to set a sum-over kernel argument.");
    }    
    
    // enqueue the kernel over all the slices (THIS IS NOT OPTIMAL)
    err = clEnqueueNDRangeKernel(_queue, sumOverKernel, 1, NULL, &slices, NULL, 0, NULL, NULL);
    if (err != CL_SUCCESS){
        cerr << "ERROR in enqueueing sum-over kernel: " << err << endl;
        throw runtime_error("Failed to enqueue sum-over kernel.");
    }
    
    // read result, sum up
    clEnqueueReadBuffer(_queue, tmp, CL_TRUE, 0, memSize, tmpHost, 0, NULL, NULL );
    if(err != CL_SUCCESS){
        cerr << "ERROR to read sum-over result" << err << endl;
        throw runtime_error("Failed to read sum-over result.");
    }
    
    double sum = 0.0;
    for(size_t x = 0; x < slices; ++x){
        sum += tmpHost[x];
    }
    
    clReleaseMemObject(tmp);
        
#ifdef LIBKEDF_DEBUG
    cout << "DEBUG: Done calling sum-over kernel in OCL grid. Result: " << sum << endl;
#endif
    
    return sum;
    
#else
    
   if(_noGridPoints % 32 != 0) {
        // needs to be 32 in a row for the fast kernel
        const size_t rows = this->_xDim;
        const size_t cols = this->_yDim;
        const size_t slices = this->_zDim;
        const cl_int sliceLength = rows*cols;
        const size_t memSize = slices*sizeof(double);
    
        double tmpHost[slices];
    
        // allocate tmp scratch
        cl_int err;
        cl_mem tmp = clCreateBuffer(_ctx, CL_MEM_WRITE_ONLY, memSize, NULL, &err);
        if(err != CL_SUCCESS){
            cerr << "ERROR in allocating GPU memory of size " << memSize << " with error " << err << endl;
            throw runtime_error("Failed to allocate GPU memory");
        }
    
        err  = clSetKernelArg(sumOverKernel, 0, sizeof(cl_mem), &_bufReal);
        err |= clSetKernelArg(sumOverKernel, 1, sizeof(cl_mem), &tmp);
        err |= clSetKernelArg(sumOverKernel, 2, sizeof(int), &sliceLength);
        if(err != CL_SUCCESS){
            cerr << "ERROR in setting a sum-over kernel argument: " << err << endl;
            throw runtime_error("Failed to set a sum-over kernel argument.");
        }    
    
        // enqueue the kernel over all the slices (THIS IS NOT OPTIMAL)
        err = clEnqueueNDRangeKernel(_queue, sumOverKernel, 1, NULL, &slices, NULL, 0, NULL, NULL);
        if (err != CL_SUCCESS){
            cerr << "ERROR in enqueueing sum-over kernel: " << err << endl;
            throw runtime_error("Failed to enqueue sum-over kernel.");
        }
    
        // read result, sum up
        clEnqueueReadBuffer(_queue, tmp, CL_TRUE, 0, memSize, tmpHost, 0, NULL, NULL );
        if(err != CL_SUCCESS){
            cerr << "ERROR to read sum-over result" << err << endl;
            throw runtime_error("Failed to read sum-over result.");
        }
    
        double sum = 0.0;
        for(size_t x = 0; x < slices; ++x){
            sum += tmpHost[x];
        }
    
        clReleaseMemObject(tmp);
        
        return sum;
    }
    
    cl_int err;
    
    const size_t global_size = _noGridPoints / 8; // double4, summation of two
    const size_t local_size = 4;
    const cl_int num_groups = global_size/local_size;
    const double* sum = new double[num_groups];
   
    cl_mem sum_buffer = clCreateBuffer(_ctx, CL_MEM_READ_WRITE |
        CL_MEM_COPY_HOST_PTR, num_groups * sizeof(double), (void*)sum, &err);
    if(err < 0) {
        cerr << "ERROR Couldn't create temporary buffer for addUp kernel: " << err << endl;
        throw runtime_error("Couldn't create temporary buffer for addUp kernel.");
    }
    
    err = clSetKernelArg(addUpKernel, 0, sizeof(cl_mem), &_bufReal);
    err |= clSetKernelArg(addUpKernel, 1, local_size * sizeof(double), NULL);
    err |= clSetKernelArg(addUpKernel, 2, sizeof(cl_mem), &sum_buffer);
    if(err < 0) {
        cerr << "ERROR Couldn't set an argument for addUp kernel: " << err << endl;
        throw runtime_error("Failed to set an argument for addUp kernel.");
    }
    err = clEnqueueNDRangeKernel(_queue, addUpKernel, 1, NULL, &global_size,
        &local_size, 0, NULL, NULL);
    if(err < 0) {
        cerr << "ERROR in enqueueing addUp kernel: " << err << endl;
        throw runtime_error("Failed to enqueue addUp kernel.");
    }
    err = clEnqueueReadBuffer(_queue, sum_buffer, CL_TRUE, 0,
         sizeof(double)*num_groups, (void*) sum, 0, NULL, NULL);
    if(err < 0) {
        cerr << "ERROR in reading result buffer for addUp kernel: " << err << endl;
        throw runtime_error("Failed to read result buffer for addUp kernel.");
    }
   
    double total = 0.0;
    for(int j = 0; j < num_groups; j++) {
        total += sum[j];
    }
   
    clReleaseMemObject(sum_buffer);
    delete[] sum;
   
   return total;
#endif
}

void CartesianOCLOOPGrid::minMax(double& min, double& max){
    
#ifdef LIBKEDF_DEBUG
    cout << "DEBUG: Calling min/max kernel in OCL grid." << endl;
#endif
    
    if(!_realOnGPUInSync){        
        if(!_realInSync){
            enqueueBackwardTransform();
        } else {
            transferRealToGPU();
        }
    }
        
    if(_noGridPoints % 32 != 0) {
        // needs to be 32 in a row for the fast kernel
        const size_t rows = this->_xDim;
        const size_t cols = this->_yDim;
        const size_t slices = this->_zDim;
        const cl_int sliceLength = rows*cols;
        const size_t memSize = slices*sizeof(double);
    
        double tmpMinHost[slices];
        double tmpMaxHost[slices];
    
        // allocate tmp scratch        
        cl_int err;
        cl_mem tmpMin = clCreateBuffer(_ctx, CL_MEM_WRITE_ONLY, memSize, NULL, &err);
        cl_mem tmpMax = clCreateBuffer(_ctx, CL_MEM_WRITE_ONLY, memSize, NULL, &err);
        if(err != CL_SUCCESS){
            cerr << "ERROR in allocating GPU memory of size " << memSize << " with error " << err << endl;
            throw runtime_error("Failed to allocate GPU memory");
        }
    
        err  = clSetKernelArg(minMaxGridKernel, 0, sizeof(cl_mem), &_bufReal);
        err |= clSetKernelArg(minMaxGridKernel, 1, sizeof(cl_mem), &tmpMin);
        err |= clSetKernelArg(minMaxGridKernel, 2, sizeof(cl_mem), &tmpMax);
        err |= clSetKernelArg(minMaxGridKernel, 3, sizeof(int), &sliceLength);
        if(err != CL_SUCCESS){
            cerr << "ERROR in setting a min/max kernel argument: " << err << endl;
            throw runtime_error("Failed to set a min/max kernel argument.");
        }    
    
        // enqueue the kernel over all the slices (THIS IS NOT OPTIMAL)
        err = clEnqueueNDRangeKernel(_queue, minMaxGridKernel, 1, NULL, &slices, NULL, 0, NULL, NULL);
        if (err != CL_SUCCESS){
            cerr << "ERROR in enqueueing min/max kernel: " << err << endl;
            throw runtime_error("Failed to enqueue min/max kernel.");
        }
    
        // read result, sum up
        clEnqueueReadBuffer(_queue, tmpMin, CL_FALSE, 0, memSize, tmpMinHost, 0, NULL, NULL );
        if(err != CL_SUCCESS){
            cerr << "ERROR to read min result" << err << endl;
            throw runtime_error("Failed to read min result.");
        }
        clEnqueueReadBuffer(_queue, tmpMax, CL_TRUE, 0, memSize, tmpMaxHost, 0, NULL, NULL );
        if(err != CL_SUCCESS){
            cerr << "ERROR to read max result" << err << endl;
            throw runtime_error("Failed to read max result.");
        }
    
        double myMin = tmpMinHost[0];
        double myMax = tmpMaxHost[0];
        for(size_t x = 1; x < slices; ++x){
            myMin = std::min(myMin, tmpMinHost[x]);
            myMax = std::max(myMax, tmpMaxHost[x]);
        }
        
        clReleaseMemObject(tmpMin);
        clReleaseMemObject(tmpMax);

        min = myMin;
        max = myMax;
        
        return;
    }
        
    cl_int err;
    
    const size_t global_size = _noGridPoints / 8; // double4, summation of two
    const size_t local_size = 4;
    const cl_int num_groups = global_size/local_size;
    const double* minArr = new double[num_groups];
    const double* maxArr = new double[num_groups];
   
    cl_mem min_buffer = clCreateBuffer(_ctx, CL_MEM_READ_WRITE |
        CL_MEM_COPY_HOST_PTR, num_groups * sizeof(double), (void*)minArr, &err);
    if(err < 0) {
        cerr << "ERROR Couldn't create 1st temporary buffer for min/max fast kernel: " << err << endl;
        throw runtime_error("Couldn't create 1st temporary buffer for min/max fast kernel.");
    }
    cl_mem max_buffer = clCreateBuffer(_ctx, CL_MEM_READ_WRITE |
        CL_MEM_COPY_HOST_PTR, num_groups * sizeof(double), (void*)maxArr, &err);
    if(err < 0) {
        cerr << "ERROR Couldn't create 2nd temporary buffer for min/max fast kernel: " << err << endl;
        throw runtime_error("Couldn't create 2nd temporary buffer for min/max fast kernel.");
    }
    
    err = clSetKernelArg(minMaxFastKernel, 0, sizeof(cl_mem), &_bufReal);
    err |= clSetKernelArg(minMaxFastKernel, 1, local_size * sizeof(double), NULL);
    err |= clSetKernelArg(minMaxFastKernel, 2, sizeof(cl_mem), &min_buffer);
    err |= clSetKernelArg(minMaxFastKernel, 3, local_size * sizeof(double), NULL);
    err |= clSetKernelArg(minMaxFastKernel, 4, sizeof(cl_mem), &max_buffer);
    if(err < 0) {
        cerr << "ERROR Couldn't set an argument for min/max fast kernel: " << err << endl;
        throw runtime_error("Failed to set an argument for min/max fast kernel.");
    }
    err = clEnqueueNDRangeKernel(_queue, minMaxFastKernel, 1, NULL, &global_size,
        &local_size, 0, NULL, NULL);
    if(err < 0) {
        cerr << "ERROR in enqueueing min/max fast kernel: " << err << endl;
        throw runtime_error("Failed to enqueue min/max fast kernel.");
    }
    err = clEnqueueReadBuffer(_queue, min_buffer, CL_FALSE, 0,
         sizeof(double)*num_groups, (void*) minArr, 0, NULL, NULL);
    err |= clEnqueueReadBuffer(_queue, max_buffer, CL_TRUE, 0,
         sizeof(double)*num_groups, (void*) maxArr, 0, NULL, NULL);
    if(err < 0) {
        cerr << "ERROR in reading result buffers for min/max fast kernel: " << err << endl;
        throw runtime_error("Failed to read result buffers for min/max fast kernel.");
    }
   
    double myMin = minArr[0];
    double myMax = maxArr[0];
    for(int j = 1; j < num_groups; j++) {
        myMin = std::min(myMin, minArr[j]);
        myMax = std::max(myMax, maxArr[j]);
    }
   
    clReleaseMemObject(min_buffer);
    clReleaseMemObject(max_buffer);
    delete[] minArr;
    delete[] maxArr;

    min = myMin;
    max = myMax;
}

void CartesianOCLOOPGrid::transferRealToGPU(){
    
    // check if GPU memory needs to be allocated
    allocateGPUMemory(REAL);

    if(!_realOnGPUInSync){
        if(!_realInSync){
            enqueueBackwardTransform();
        } else {
            cl_int err;
            
#ifdef LIBKEDF_DEBUG
            cout << "DEBUG: transferring real grid to the GPU" << endl;
            cout << "       queue: " << _queue << " Blocking? " << CL_FALSE << " size: " << _bufferSizeHost << endl;
#endif
            // copy real buffer to GPU
            err = clEnqueueWriteBuffer(_queue, _bufReal, CL_FALSE, 0, _bufferSizeHost, _hostMem, 0, NULL, NULL );
            if(err != CL_SUCCESS){
                cerr << "ERROR to write real data to GPU " << err << endl;
                throw runtime_error("Failed to write real data to GPU.");
            }
            _realOnGPUInSync = true;
            _reciOnGPUInSync = false;
        }
    }
}

void CartesianOCLOOPGrid::transferReciToGPU(){
    
    // check if GPU memory needs to be allocated
    allocateGPUMemory(RECI);

    if(!_reciOnGPUInSync){
        if(!_reciInSync){
            enqueueForwardTransform();
        } else {
            cl_int err;
        
#ifdef LIBKEDF_DEBUG
            cout << "DEBUG: transferring reciprocal grid to the GPU" << endl;
#endif
            // copy reciprocal buffer to GPU
            err = clEnqueueWriteBuffer(_queue, _bufReci, CL_FALSE, 0, _bufferSizeHost, _hostMem, 0, NULL, NULL );
            if(err != CL_SUCCESS){
                cerr << "ERROR to write reciprocal data to GPU " << err << endl;
                throw runtime_error("Failed to write reciprocal data to GPU.");
            }
            _reciOnGPUInSync = true;
            _realOnGPUInSync = false;
        }
    }
}

void CartesianOCLOOPGrid::transferRealFromGPU(){
    
#ifdef LIBKEDF_DEBUG
            cout << "DEBUG: transferring real grid from the GPU" << endl;
            cout << "       queue: " << _queue << " Blocking? " << CL_FALSE << " size: " << _bufferSizeHost << endl;
#endif
            
    // Fetch results of calculations.
    cl_int err = clEnqueueReadBuffer(_queue, _bufReal, CL_TRUE, 0, _bufferSizeHost, _hostMem, 0, NULL, NULL );
    if(err != CL_SUCCESS){
        cerr << "ERROR to read real result" << err << endl;
        throw runtime_error("Failed to real result.");
    }
    
    _realInSync = true;
    _realOnGPUInSync = true;    
}

void CartesianOCLOOPGrid::transferReciFromGPU(){
    
#ifdef LIBKEDF_DEBUG
            cout << "DEBUG: transferring reciprocal grid from the GPU" << endl;
            cout << "       queue: " << _queue << " Blocking? " << CL_FALSE << " size: " << _bufferSizeHost << endl;
#endif
    
    // Fetch results of calculations.
    cl_int err = clEnqueueReadBuffer(_queue, _bufReci, CL_TRUE, 0, _bufferSizeHost, _hostMem, 0, NULL, NULL );
    if(err != CL_SUCCESS){
        cerr << "ERROR to read reciprocal result" << err << endl;
        throw runtime_error("Failed to reciprocal result.");
    }
    
    _reciInSync = true;
    _reciOnGPUInSync = true;
}

void CartesianOCLOOPGrid::enqueueForwardTransform(){

#ifdef LIBKEDF_DEBUG
    cout << "DEBUG: Enqueue of r2c FFT in OCL grid started." << endl;
#endif
    
    if(!_realOnGPUInSync){
      _realInSync = true; // to avoid this being c2r'd -> r2c'd -> c2r'd ....
    }

    allocateGPUMemory(ALL);
    this->transferRealToGPU();
        
    cl_int err;
        
#ifdef LIBKEDF_DEBUG
    cout << "DEBUG: doing r2c FFT in OCL grid." << endl;
#endif

    err = clfftEnqueueTransform(*_r2c_plan, CLFFT_FORWARD, 1, &_queue, 0, NULL, NULL, &_bufReal, &_bufReci, NULL);
    if(err != CL_SUCCESS){
        cerr << "ERROR to enqueue r2c transform " << err << endl;
        throw runtime_error("Failed to enqueue r2c transform.");
    }

    _realOnGPUInSync = true;
    _reciOnGPUInSync = true;
    _reciInSync = false; // not transferred yet
}

void CartesianOCLOOPGrid::enqueueBackwardTransform(){
    
    allocateGPUMemory(ALL);
    this->transferReciToGPU();
        
    cl_int err;
        
#ifdef LIBKEDF_DEBUG
    cout << "DEBUG: doing c2r FFT in OCL grid." << endl;
#endif
        
    err = clfftEnqueueTransform(*_c2r_plan, CLFFT_BACKWARD, 1, &_queue, 0, NULL, NULL, &_bufReci, &_bufReal, NULL);
    if(err != CL_SUCCESS){
        cerr << "ERROR to enqueue c2r transform " << err << endl;
        throw runtime_error("Failed to enqueue c2r transform.");
    }
    
    _realOnGPUInSync = true;
    _reciOnGPUInSync = true;
    _realInSync = false; // not transferred yet
}

cl_mem CartesianOCLOOPGrid::getRealGPUBuffer(){
    _realInSync = false; // this is writable!
    if(!_bufReal){
        // lazy allocation
        allocateGPUMemory(REAL);
    }
    return _bufReal;
}

cl_mem CartesianOCLOOPGrid::getReciGPUBuffer(){
    _realInSync = false; // this is writable!
    if(!_bufReci){
        // lazy allocation
        allocateGPUMemory(RECI);
    }
    return _bufReci;
}

const cl_mem CartesianOCLOOPGrid::readRealGPUBuffer() const {
    if(!_bufReal){
        cerr << "ERROR: Can't read real buffer if not allocated before!" << endl;
        throw runtime_error("Can't read recal buffer, not allocated.");
    }
    return _bufReal;
}

const cl_mem CartesianOCLOOPGrid::readReciGPUBuffer() const {
    if(!_bufReci){
        cerr << "ERROR: Can't read reciprocal buffer if not allocated before!" << endl;
        throw runtime_error("Can't read reciprocal buffer, not allocated.");
    }
    return _bufReci;
}

cl_context CartesianOCLOOPGrid::getGPUContext(){
    return _ctx;
}

cl_command_queue CartesianOCLOOPGrid::getGPUQueue(){
    return _queue;
}

cl_device_id* CartesianOCLOOPGrid::getGPUDevices(){
    return _devices.get();
}

cl_uint CartesianOCLOOPGrid::getNoGPUDevices(){
    return 1; // only single GPU for the time being
}

void CartesianOCLOOPGrid::markRealDirty(){
    _realInSync = false;
}

void CartesianOCLOOPGrid::markReciDirty(){
    _reciInSync = false;
}

void CartesianOCLOOPGrid::markRealClean(){
    _realInSync = true;
}

void CartesianOCLOOPGrid::markReciClean(){
    _reciInSync = true;
}

void CartesianOCLOOPGrid::markRealGPUDirty(){
    _realOnGPUInSync = false;
}

void CartesianOCLOOPGrid::markReciGPUDirty(){
    _reciOnGPUInSync = false;
}

void CartesianOCLOOPGrid::markRealGPUClean(){
    _realOnGPUInSync = true;
}

void CartesianOCLOOPGrid::markReciGPUClean(){
    _reciOnGPUInSync = true;
}

size_t CartesianOCLOOPGrid::getVectortypeAlignment() const {
    
#ifdef LIBKEDF_HARDCODE_OCLVECTORSIZE
    return LIBKEDF_OCLVECTORSIZE;
#else
    // compute maximum alignment (in real space!)
    if(_noGridPoints % 16 == 0){
        return 16;
    } else if(_noGridPoints % 8 == 0){
        return 8;
    } else if(_noGridPoints % 4 == 0){
        return 4;
    } else if(_noGridPoints % 2 == 0){
        return 2;
    }
    
    return 1;
#endif    
}

const char* CartesianOCLOOPGrid::getCompilationOptions() const {
    
    // XXX this is leaking a bit of memory ATM
    ostringstream os;
    os << COMPILATIONOPTS;
    os << " -D KEDFOCLMODE=1 ";
    
    const string opts = os.str();
    
#ifdef LIBKEDF_DEBUG
    cout << "DEBUG: OpenCL compilation options are " << opts << endl;
#endif
    
    return opts.c_str();
}

const string CartesianOCLOOPGrid::getMacroDefinitions() const {
    
    const size_t align = getVectortypeAlignment();
    
#ifdef LIBKEDF_HAS_POWR
    const string powr = "powr";
#else
    const string powr = "pow";
#endif
    
    const string alType = "#define KEDFOCLV double";
    ostringstream os;
    os << alType;
    if(align != 1){
    	os << align;
    }
    os << "\n";
    
    os << "#define KEDFPOWR ";
    os << powr;
    os << "\n";
    
    const string enable64 = "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";
    os << enable64;
    
    return os.str();
}

void CartesianOCLOOPGrid::finalize(){
   
    cl_int err = clFinish(_queue);
    if(err != CL_SUCCESS){
        cerr << "ERROR to waiting for calculations " << err << endl;
        throw runtime_error("Failed to wait for calculations.");
    }
}

void CartesianOCLOOPGrid::updateRealGrid(const double* rawData){
    
    allocateGPUMemory(REAL);
    
    // directly put this on the GPU
    
    _realInSync = false; // not transferred yet
    _reciInSync = false;
    _realOnGPUInSync = true;
    _reciOnGPUInSync = false;
    
    const size_t nSlices = this->getGridPointsZ();
    const size_t nRows = this->getGridPointsY();
    const size_t nCols = this->getGridPointsX();
    const size_t totSize = nSlices*nRows*nCols*sizeof(double);
    
    cl_int err = clEnqueueWriteBuffer(_queue, _bufReal, CL_FALSE, 0, totSize, rawData, 0, NULL, NULL );
    if(err != CL_SUCCESS){
        cerr << "ERROR to write update data on GPU " << err << endl;
        throw runtime_error("Failed to update real data on GPU.");
    }
}
    
void CartesianOCLOOPGrid::getRealGridData(double* rawData){
    
    finalize(); // drain queues
    
    const size_t nSlices = this->getGridPointsZ();
    const size_t nRows = this->getGridPointsY();
    const size_t nCols = this->getGridPointsX();
    const size_t totSize = nSlices*nRows*nCols*sizeof(double);
    
    if(_realInSync){
        
        memcpy(rawData, this->_hostMem, totSize);
        
        return;
    }
    
    if(!_realOnGPUInSync){
        this->enqueueBackwardTransform();
    }
    
    cl_int err = clEnqueueReadBuffer(_queue, _bufReal, CL_TRUE, 0, totSize, rawData, 0, NULL, NULL );
    if(err != CL_SUCCESS){
        cerr << "ERROR to read real grid data " << err << endl;
        throw runtime_error("Failed to real grid data.");
    }
}
