/* 
 * Author: Johannes M Dieterich
 */

#ifndef CARTESIANOCLOOPGRID_HPP
#define	CARTESIANOCLOOPGRID_HPP

#include <CL/cl.h>
#include <clFFT.h>
#include <string>
#include "CartesianGrid.hpp"
using namespace std;

enum gpumem_allocation_t {NONE, REAL, RECI, ALL};

class CartesianOCLOOPGrid: public CartesianGrid {
public:
    CartesianOCLOOPGrid(const size_t rows, const size_t cols, const size_t slices, const shared_ptr<mat> cellVectors, const size_t platformNo, const size_t deviceNo);
    ~CartesianOCLOOPGrid();
    
    unique_ptr<CartesianOCLOOPGrid> duplicate() const;
    unique_ptr<CartesianOCLOOPGrid> emptyDuplicate() const;
    
    void addGrid(CartesianOCLOOPGrid *other);
    void fmaGrid(const double mult, CartesianOCLOOPGrid *other);
    
    unique_ptr<FourierGrid> createFourierDuplicate() const override;
    unique_ptr<FourierGrid> createFourierEmptyDuplicate() const override;
    
    arma::cube* getRealGrid() override;
    arma::cx_cube* getReciprocalGrid() override;
    
    void complete(arma::cube* realGrid) override;
    void completeReciprocal(arma::cx_cube* reciprocalGrid) override;
    
    void resetToReal() override;
    void resetToReciprocal() override;
    
    const arma::cube* readRealGrid() override;
    const arma::cube* tryReadRealGrid() const override;
    const arma::cx_cube* readReciprocalGrid() override;
    
    void multiplyGNorms() override;
    
    void sqrtGrid() override;
    
    void powGrid(const double exponent) override;
    
    void transferRealToGPU();
    void transferReciToGPU();
    void transferRealFromGPU();
    void transferReciFromGPU();
    void enqueueForwardTransform();
    void enqueueBackwardTransform();
    
    size_t getVectortypeAlignment() const;
    
    void allocateGPUMemory(const gpumem_allocation_t alloc); //XXX move to private !!!
    
    unique_ptr<CartesianOCLOOPGrid> laplacian();
    
    unique_ptr<CartesianOCLOOPGrid> gradientSquared();
    
    void multiplyGVectorsX();
    
    void multiplyGVectorsY();
    
    void multiplyGVectorsZ();
    
    double sumOver() override;
    
    void minMax(double& min, double& max) override;
    
    unique_ptr<CartesianOCLOOPGrid> directionalDivergenceX(){

#ifdef LIBKEDF_DEBUG
        cout << "DEBUG: doing directional divergence calculation X in Cartesian grid." << endl;
#endif
        
        unique_ptr<CartesianOCLOOPGrid> divergenceG(this->duplicate());
        divergenceG->multiplyGVectorsX();
        
        return divergenceG;
    }
    
    unique_ptr<CartesianOCLOOPGrid> directionalDivergenceY(){
        
#ifdef LIBKEDF_DEBUG
        cout << "DEBUG: doing directional divergence calculation Y in Cartesian grid." << endl;
#endif
        unique_ptr<CartesianOCLOOPGrid> divergenceG(this->duplicate());
        divergenceG->multiplyGVectorsY();
        
        return divergenceG;
    }
    
    unique_ptr<CartesianOCLOOPGrid> directionalDivergenceZ(){
        
#ifdef LIBKEDF_DEBUG
        cout << "DEBUG: doing directional divergence calculation Z in Cartesian grid." << endl;
#endif
        unique_ptr<CartesianOCLOOPGrid> divergenceG(this->duplicate());
        divergenceG->multiplyGVectorsZ();
        
        return divergenceG;
    }
    
    void multiplyElementwise(CartesianOCLOOPGrid* grid);
    
    void multiplyTwoSqrtOf(CartesianOCLOOPGrid* other);
    
    cl_mem getRealGPUBuffer();
    cl_mem getReciGPUBuffer();
    
    const cl_mem readRealGPUBuffer() const;
    const cl_mem readReciGPUBuffer() const;
    
    cl_context getGPUContext();
    cl_command_queue getGPUQueue();
    cl_device_id* getGPUDevices();
    cl_uint getNoGPUDevices();
    
    cl_kernel getComplexMultKernel(){
        return clCreateKernel(cartOCLOOCPProgram, "cmplMultGrid", NULL);
    }
    
    cl_kernel getRealComplexMultKernel(){
        return clCreateKernel(cartOCLOOCPProgram, "cmplRealMultGrid", NULL);
    }

    cl_kernel getRealMultKernel(){
        return clCreateKernel(cartOCLOOCPProgram, "realMultGrid", NULL);
    }
    
    void markRealDirty();
    void markReciDirty();
    
    void markRealClean();
    void markReciClean();
    
    void markRealGPUDirty();
    void markReciGPUDirty();
    
    void markRealGPUClean();
    void markReciGPUClean();
    
    const char* getCompilationOptions() const;
    
    const string getMacroDefinitions() const;
    
    void finalize() override;
    
    void updateRealGrid(const double* rawData) override;

    void getRealGridData(double* rawData) override;
    
    void transferGNorms();

private:
    CartesianOCLOOPGrid(const CartesianOCLOOPGrid& orig);
    
    //void allocateGPUMemory(); XXX!!!!
    void freeGPUMemory(const gpumem_allocation_t alloc);
    
    void multiplyGVectors(const cube* vectors);
    
    bool _realInSync;
    bool _reciInSync;
    bool _realOnGPUInSync;
    bool _reciOnGPUInSync;
    bool _realReturned;
    bool _reciReturned;
    size_t _bufferSizeHost;
    void* _hostMem;
    shared_ptr<cl_platform_id> _platforms;
    cl_uint _noPlatforms;
    shared_ptr<cl_device_id> _devices;
    cl_uint _noDevices;
    cl_context _ctx;
    cl_command_queue _queue;
    shared_ptr<clfftPlanHandle> _r2c_plan;
    shared_ptr<clfftPlanHandle> _c2r_plan;
    cl_mem _bufReal;
    cl_mem _bufReci;
    
#ifndef LIBKEDF_LOWGPUMEMORY
    cl_mem _gNormsBuf = NULL;
#endif

    cl_program cartOCLOOCPProgram;
    
    cl_kernel realPrintKernel;
    cl_kernel complexPrintKernel;
    
    cl_kernel addKernel;
    cl_kernel fmaKernel;
    
    cl_kernel sqrtKernel;
    size_t _localSqrtSize;
    
    cl_kernel powKernel;
    size_t _localPowSize;
        
    cl_kernel realMultKernel;
    size_t _localRealMultSize;
    
    cl_kernel realAddSq0Kernel;
    size_t _localRealAddSq0Size;
    
    cl_kernel realAddSqKernel;
    size_t _localRealAddSqSize;
    
    cl_kernel cmplMultKernel;
    size_t _localCmplMultSize;
    
    cl_kernel cmplRealMultKernel;
    size_t _localCmplRealMultSize;
    
    cl_kernel pureCmplMultKernel;    
    
    cl_kernel gNormMultKernel;
    size_t _localGNormMultSize;
    
    cl_kernel sumOverKernel;
    cl_kernel addUpKernel;
    size_t _localSumOverSize;
    
    cl_kernel minMaxFastKernel;
    cl_kernel minMaxGridKernel;
    
    cl_kernel mult2SqrtKernel;
    
    arma::cube* _realGrid;
    arma::cx_cube* _reciGrid;
    
    const static string COMPILATIONOPTS;
};

#endif	/* CARTESIANOCLOOPGRID_HPP */

