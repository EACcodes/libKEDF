/* 
 * Copyright (c) 2016-2017, Princeton University, Johannes M Dieterich, Emily A Carter
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
#include <stdlib.h>
#include <memory>
#include <sstream>
#include "libKEDF.h"
#include "CartesianOOPGrid.hpp"
#include "KEDF.hpp"
#include "KEDFFactory.hpp"
#include "MultiTermKEDF.hpp"
#ifdef LIBKEDF_OCL
#include "CartesianOCLOOPGrid.hpp"
#include <clFFT.h>
#endif
using namespace std;
using namespace arma;

#ifdef __cplusplus
extern "C" {
#endif

struct libkedf_data* libkedf_init_(){
    return (libkedf_data*) calloc(1,sizeof(libkedf_data));
}
    
void libkedf_initialize_grid_(struct libkedf_data *dat, const int *x, const int *y, const int *z,
        const double *vecX, const double *vecY, const double *vecZ){

    dat->configuredGridType = SMP;
    
    shared_ptr<mat> cellVectors = make_shared<mat>(3,3);
    cellVectors->at(0,0) = vecX[0];
    cellVectors->at(0,1) = vecX[1];
    cellVectors->at(0,2) = vecX[2];
    
    cellVectors->at(1,0) = vecY[0];
    cellVectors->at(1,1) = vecY[1];
    cellVectors->at(1,2) = vecY[2];
    
    cellVectors->at(2,0) = vecZ[0];
    cellVectors->at(2,1) = vecZ[1];
    cellVectors->at(2,2) = vecZ[2];
    
    dat->grid = new CartesianOOPGrid(*x,*y,*z,cellVectors);
}

#ifdef LIBKEDF_OCL
void libkedf_initialize_grid_ocl_(struct libkedf_data *dat, const int *x, const int *y, const int *z,
        const double *vecX, const double *vecY, const double *vecZ, const int *platformNo,
        const int *deviceNo){
    
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
    
    dat->configuredGridType = OCL;
    
    shared_ptr<mat> cellVectors = make_shared<mat>(3,3);
    cellVectors->at(0,0) = vecX[0];
    cellVectors->at(0,1) = vecX[1];
    cellVectors->at(0,2) = vecX[2];
    
    cellVectors->at(1,0) = vecY[0];
    cellVectors->at(1,1) = vecY[1];
    cellVectors->at(1,2) = vecY[2];
    
    cellVectors->at(2,0) = vecZ[0];
    cellVectors->at(2,1) = vecZ[1];
    cellVectors->at(2,2) = vecZ[2];
    
    const size_t myPlat = *platformNo;
    const size_t myDev = *deviceNo;
    dat->gridOCL = new CartesianOCLOOPGrid(*x,*y,*z,cellVectors,myPlat,myDev);
}
#endif

void libkedf_update_cellvectors_(struct libkedf_data *dat, const double *vecX, const double *vecY, const double *vecZ){
    
    switch(dat->configuredGridType){
        case SMP:
            dat->grid->updateCellVectors(vecX, vecY, vecZ);
            break;
#ifdef LIBKEDF_OCL            
        case OCL:
            dat->gridOCL->updateCellVectors(vecX, vecY, vecZ);
            break;
#endif
        default:
            cerr << "Unknown case in libkedf_update_cellvectors_" << endl;
    }
}
    
void libkedf_initialize_tf_(struct libkedf_data *dat){
    
    const string config = "ThomasFermi";
    
    if(dat->configuredGridType == SMP){
        KEDFFactory<CartesianOOPGrid>* factory = new KEDFFactory<CartesianOOPGrid>();
        dat->kedf = factory->constructKEDF(dat->grid,config);
        delete factory;
    }
#ifdef LIBKEDF_OCL     
    else if(dat->configuredGridType == OCL){
        KEDFFactory<CartesianOCLOOPGrid>* factory = new KEDFFactory<CartesianOCLOOPGrid>();
        dat->kedfOCL = factory->constructKEDF(dat->gridOCL,config);
        delete factory;
    }
#endif
    else {
        cerr << "Unknown case in libkedf_initialize_tf_" << endl;
    }
}
    
void libkedf_initialize_vw_(struct libkedf_data *dat){
    
    const string config = "vonWeizsaecker";
    
    if(dat->configuredGridType == SMP){
        KEDFFactory<CartesianOOPGrid>* factory = new KEDFFactory<CartesianOOPGrid>();
        dat->kedf = factory->constructKEDF(dat->grid,config);
        delete factory;
    }
#ifdef LIBKEDF_OCL     
    else if(dat->configuredGridType == OCL){
        KEDFFactory<CartesianOCLOOPGrid>* factory = new KEDFFactory<CartesianOCLOOPGrid>();
        dat->kedfOCL = factory->constructKEDF(dat->gridOCL,config);
        delete factory;
    }
#endif
    else {
        cerr << "Unknown case in libkedf_initialize_vw_" << endl;
    }
}
    
void libkedf_initialize_tf_plus_vw_(struct libkedf_data *dat, const double *a, const double *b){
    
    const string configTF = "ThomasFermi";
    const string configvW = "vonWeizsaecker";
    
    if(dat->configuredGridType == SMP){
        KEDFFactory<CartesianOOPGrid>* factory = new KEDFFactory<CartesianOOPGrid>();
    
        unique_ptr<vector<double> > weights = make_unique<vector<double> >();
        unique_ptr<vector<shared_ptr<KEDF<CartesianOOPGrid> > > > kedfs = make_unique<vector<shared_ptr<KEDF<CartesianOOPGrid> > > >();
    
        KEDF<CartesianOOPGrid> *tf = factory->constructKEDF(dat->grid,configTF);
        KEDF<CartesianOOPGrid> *vW = factory->constructKEDF(dat->grid,configvW);
    
        shared_ptr<KEDF<CartesianOOPGrid> > tfSh(tf);
        shared_ptr<KEDF<CartesianOOPGrid> > vWSh(vW);
    
        // a*TF+b*vW
        weights->push_back(*a);
        kedfs->push_back(tfSh);
        weights->push_back(*b);
        kedfs->push_back(vWSh);
    
        dat->kedf = new MultiTermKEDF<CartesianOOPGrid>(dat->grid, move(kedfs), move(weights));
    
        delete factory;
    }
#ifdef LIBKEDF_OCL
    else if(dat->configuredGridType == OCL){
        KEDFFactory<CartesianOCLOOPGrid>* factory = new KEDFFactory<CartesianOCLOOPGrid>();
    
        unique_ptr<vector<double> > weights = make_unique<vector<double> >();
        unique_ptr<vector<shared_ptr<KEDF<CartesianOCLOOPGrid> > > > kedfs = make_unique<vector<shared_ptr<KEDF<CartesianOCLOOPGrid> > > >();
    
        KEDF<CartesianOCLOOPGrid> *tf = factory->constructKEDF(dat->gridOCL,configTF);
        KEDF<CartesianOCLOOPGrid> *vW = factory->constructKEDF(dat->gridOCL,configvW);
    
        shared_ptr<KEDF<CartesianOCLOOPGrid> > tfSh(tf);
        shared_ptr<KEDF<CartesianOCLOOPGrid> > vWSh(vW);
    
        // a*TF+b*vW
        weights->push_back(*a);
        kedfs->push_back(tfSh);
        weights->push_back(*b);
        kedfs->push_back(vWSh);
    
        dat->kedfOCL = new MultiTermKEDF<CartesianOCLOOPGrid>(dat->gridOCL, move(kedfs), move(weights));
    
        delete factory;
    }
#endif
    else{
        cerr << "Unknown case in libkedf_initialize_tf_plus_vw_" << endl;
    }
}

void libkedf_initialize_wt_(struct libkedf_data *dat, const double *rho0){
    
    const string configTF = "ThomasFermi";
    const string configvW = "vonWeizsaecker";
    
    if(dat->configuredGridType == SMP){
        KEDFFactory<CartesianOOPGrid>* factory = new KEDFFactory<CartesianOOPGrid>();
    
        ostringstream oss;
        oss << "WangTeter: rho0= ";
        oss << (*rho0);
    
        const string configWT = oss.str();
    
        unique_ptr<vector<double> > weights = make_unique<vector<double> >();
        unique_ptr<vector<shared_ptr<KEDF<CartesianOOPGrid> > > > kedfs = make_unique<vector<shared_ptr<KEDF<CartesianOOPGrid> > > >();
    
        KEDF<CartesianOOPGrid> *tf = factory->constructKEDF(dat->grid,configTF);
        KEDF<CartesianOOPGrid> *vW = factory->constructKEDF(dat->grid,configvW);
        KEDF<CartesianOOPGrid> *wt = factory->constructFourierKEDF(dat->grid,configWT);
    
        shared_ptr<KEDF<CartesianOOPGrid> > tfSh(tf);
        shared_ptr<KEDF<CartesianOOPGrid> > vWSh(vW);
        shared_ptr<KEDF<CartesianOOPGrid> > wtSh(wt);
    
        weights->push_back(1.0);
        kedfs->push_back(tfSh);
        weights->push_back(1.0);
        kedfs->push_back(vWSh);
        weights->push_back(1.0);
        kedfs->push_back(wtSh);
    
        dat->kedf = new MultiTermKEDF<CartesianOOPGrid>(dat->grid, move(kedfs), move(weights));
    
        delete factory;
    }
#ifdef LIBKEDF_OCL
    else if(dat->configuredGridType == OCL){
        KEDFFactory<CartesianOCLOOPGrid>* factory = new KEDFFactory<CartesianOCLOOPGrid>();
    
        ostringstream oss;
        oss << "WangTeter: rho0= ";
        oss << (*rho0);
    
        const string configWT = oss.str();
    
        unique_ptr<vector<double> > weights = make_unique<vector<double> >();
        unique_ptr<vector<shared_ptr<KEDF<CartesianOCLOOPGrid> > > > kedfs = make_unique<vector<shared_ptr<KEDF<CartesianOCLOOPGrid> > > >();
    
        KEDF<CartesianOCLOOPGrid> *tf = factory->constructKEDF(dat->gridOCL,configTF);
        KEDF<CartesianOCLOOPGrid> *vW = factory->constructKEDF(dat->gridOCL,configvW);
        KEDF<CartesianOCLOOPGrid> *wt = factory->constructFourierKEDF(dat->gridOCL,configWT);
    
        shared_ptr<KEDF<CartesianOCLOOPGrid> > tfSh(tf);
        shared_ptr<KEDF<CartesianOCLOOPGrid> > vWSh(vW);
        shared_ptr<KEDF<CartesianOCLOOPGrid> > wtSh(wt);
    
        weights->push_back(1.0);
        kedfs->push_back(tfSh);
        weights->push_back(1.0);
        kedfs->push_back(vWSh);
        weights->push_back(1.0);
        kedfs->push_back(wtSh);
    
        dat->kedfOCL = new MultiTermKEDF<CartesianOCLOOPGrid>(dat->gridOCL, move(kedfs), move(weights));
    
        delete factory;
    }
#endif
    else{
        cerr << "Unknown case in libkedf_initialize_wt_" << endl;
    }
}

void libkedf_initialize_wt_custom_(struct libkedf_data *dat, const double *rho0, const double *alpha, const double *beta){
    
    const string configTF = "ThomasFermi";
    const string configvW = "vonWeizsaecker";
    
    if(dat->configuredGridType == SMP){
        KEDFFactory<CartesianOOPGrid>* factory = new KEDFFactory<CartesianOOPGrid>();
        
        unique_ptr<vector<double> > weights = make_unique<vector<double> >();
        unique_ptr<vector<shared_ptr<KEDF<CartesianOOPGrid> > > > kedfs = make_unique<vector<shared_ptr<KEDF<CartesianOOPGrid> > > >();
    
        KEDF<CartesianOOPGrid> *tf = factory->constructKEDF(dat->grid,configTF);
        KEDF<CartesianOOPGrid> *vW = factory->constructKEDF(dat->grid,configvW);
        
        const double lambdaTF = 1.0;
        const double muVW = 1.0;
        const double ft = 5.0 / 3.0;
        
        WangTeterKernel* kernel = new WangTeterKernel(*alpha, *beta, *rho0, lambdaTF, muVW, ft);

        unique_ptr<CartesianOOPGrid> work = dat->grid->emptyDuplicate();
        cx_cube* recGrid = work->getReciprocalGrid();
        const size_t rows = recGrid->n_rows;
        const size_t cols = recGrid->n_cols;
        const size_t slices = recGrid->n_slices;

        cube* kernelCube = MemoryFunctions::allocateScratchCube(rows, cols, slices);
        const cube* gNorms = work->getGNorms();
        kernel->fillWTKernelReciprocal(kernelCube, gNorms);

        delete kernel;
        
        KEDF<CartesianOOPGrid> *wt = new WangTeter<CartesianOOPGrid>(work.get(), *alpha, *beta, kernelCube);
    
        shared_ptr<KEDF<CartesianOOPGrid> > tfSh(tf);
        shared_ptr<KEDF<CartesianOOPGrid> > vWSh(vW);
        shared_ptr<KEDF<CartesianOOPGrid> > wtSh(wt);
    
        weights->push_back(1.0);
        kedfs->push_back(tfSh);
        weights->push_back(1.0);
        kedfs->push_back(vWSh);
        weights->push_back(1.0);
        kedfs->push_back(wtSh);
    
        dat->kedf = new MultiTermKEDF<CartesianOOPGrid>(dat->grid, move(kedfs), move(weights));
    
        delete factory;
    }
#ifdef LIBKEDF_OCL
    else if(dat->configuredGridType == OCL){
        KEDFFactory<CartesianOCLOOPGrid>* factory = new KEDFFactory<CartesianOCLOOPGrid>();
    
        unique_ptr<vector<double> > weights = make_unique<vector<double> >();
        unique_ptr<vector<shared_ptr<KEDF<CartesianOCLOOPGrid> > > > kedfs = make_unique<vector<shared_ptr<KEDF<CartesianOCLOOPGrid> > > >();
    
        KEDF<CartesianOCLOOPGrid> *tf = factory->constructKEDF(dat->gridOCL,configTF);
        KEDF<CartesianOCLOOPGrid> *vW = factory->constructKEDF(dat->gridOCL,configvW);
        
        const double lambdaTF = 1.0;
        const double muVW = 1.0;
        const double ft = 5.0 / 3.0;
        
        WangTeterKernel* kernel = new WangTeterKernel(*alpha, *beta, *rho0, lambdaTF, muVW, ft);

        unique_ptr<CartesianOCLOOPGrid> work = dat->gridOCL->emptyDuplicate();
        cx_cube* recGrid = work->getReciprocalGrid();
        const size_t rows = recGrid->n_rows;
        const size_t cols = recGrid->n_cols;
        const size_t slices = recGrid->n_slices;

        cube* kernelCube = MemoryFunctions::allocateScratchCube(rows, cols, slices);
        const cube* gNorms = work->getGNorms();
        kernel->fillWTKernelReciprocal(kernelCube, gNorms);

        delete kernel;
        
        KEDF<CartesianOCLOOPGrid> *wt = new WangTeter<CartesianOCLOOPGrid>(work.get(), *alpha, *beta, kernelCube);
    
        shared_ptr<KEDF<CartesianOCLOOPGrid> > tfSh(tf);
        shared_ptr<KEDF<CartesianOCLOOPGrid> > vWSh(vW);
        shared_ptr<KEDF<CartesianOCLOOPGrid> > wtSh(wt);
    
        weights->push_back(1.0);
        kedfs->push_back(tfSh);
        weights->push_back(1.0);
        kedfs->push_back(vWSh);
        weights->push_back(1.0);
        kedfs->push_back(wtSh);
    
        dat->kedfOCL = new MultiTermKEDF<CartesianOCLOOPGrid>(dat->gridOCL, move(kedfs), move(weights));
    
        delete factory;
    }
#endif
    else{
        cerr << "Unknown case in libkedf_initialize_wt_custom_" << endl;
    }
}
    
void libkedf_initialize_sm_(struct libkedf_data *dat, const double *rho0){
    
    const string configTF = "ThomasFermi";
    const string configvW = "vonWeizsaecker";

    if(dat->configuredGridType == SMP){
        KEDFFactory<CartesianOOPGrid>* factory = new KEDFFactory<CartesianOOPGrid>();
    
        ostringstream oss;
        oss << "SmargiassiMadden: rho0= ";
        oss << (*rho0);
    
        const string configSM = oss.str();
    
        unique_ptr<vector<double> > weights = make_unique<vector<double> >();
        unique_ptr<vector<shared_ptr<KEDF<CartesianOOPGrid> > > > kedfs = make_unique<vector<shared_ptr<KEDF<CartesianOOPGrid> > > >();
    
        KEDF<CartesianOOPGrid> *tf = factory->constructKEDF(dat->grid,configTF);
        KEDF<CartesianOOPGrid> *vW = factory->constructKEDF(dat->grid,configvW);
        KEDF<CartesianOOPGrid> *sm = factory->constructFourierKEDF(dat->grid,configSM);
    
        shared_ptr<KEDF<CartesianOOPGrid> > tfSh(tf);
        shared_ptr<KEDF<CartesianOOPGrid> > vWSh(vW);
        shared_ptr<KEDF<CartesianOOPGrid> > smSh(sm);
    
        weights->push_back(1.0);
        kedfs->push_back(tfSh);
        weights->push_back(1.0);
        kedfs->push_back(vWSh);
        weights->push_back(1.0);
        kedfs->push_back(smSh);
    
        dat->kedf = new MultiTermKEDF<CartesianOOPGrid>(dat->grid, move(kedfs), move(weights));
    
        delete factory;
    }
#ifdef LIBKEDF_OCL
    else if(dat->configuredGridType == OCL){
        KEDFFactory<CartesianOCLOOPGrid>* factory = new KEDFFactory<CartesianOCLOOPGrid>();
    
        ostringstream oss;
        oss << "SmargiassiMadden: rho0= ";
        oss << (*rho0);
    
        const string configSM = oss.str();
    
        unique_ptr<vector<double> > weights = make_unique<vector<double> >();
        unique_ptr<vector<shared_ptr<KEDF<CartesianOCLOOPGrid> > > > kedfs = make_unique<vector<shared_ptr<KEDF<CartesianOCLOOPGrid> > > >();
    
        KEDF<CartesianOCLOOPGrid> *tf = factory->constructKEDF(dat->gridOCL,configTF);
        KEDF<CartesianOCLOOPGrid> *vW = factory->constructKEDF(dat->gridOCL,configvW);
        KEDF<CartesianOCLOOPGrid> *sm = factory->constructFourierKEDF(dat->gridOCL,configSM);
    
        shared_ptr<KEDF<CartesianOCLOOPGrid> > tfSh(tf);
        shared_ptr<KEDF<CartesianOCLOOPGrid> > vWSh(vW);
        shared_ptr<KEDF<CartesianOCLOOPGrid> > smSh(sm);
    
        weights->push_back(1.0);
        kedfs->push_back(tfSh);
        weights->push_back(1.0);
        kedfs->push_back(vWSh);
        weights->push_back(1.0);
        kedfs->push_back(smSh);
    
        dat->kedfOCL = new MultiTermKEDF<CartesianOCLOOPGrid>(dat->gridOCL, move(kedfs), move(weights));
    
        delete factory;
    }
#endif
    else{
        cerr << "Unknown case in libkedf_initialize_sm_" << endl;
    }
}

void libkedf_initialize_wgc1st_(struct libkedf_data *dat, const double *rho0){
    
    const string configTF = "ThomasFermi";
    const string configvW = "vonWeizsaecker";

    if(dat->configuredGridType == SMP){
        KEDFFactory<CartesianOOPGrid>* factory = new KEDFFactory<CartesianOOPGrid>();
    
        ostringstream oss;
        oss << "1st-order-WangGovindCarter99: rho0= ";
        oss << (*rho0);
    
        const string configWGC = oss.str();
    
        unique_ptr<vector<double> > weights = make_unique<vector<double> >();
        unique_ptr<vector<shared_ptr<KEDF<CartesianOOPGrid> > > > kedfs = make_unique<vector<shared_ptr<KEDF<CartesianOOPGrid> > > >();
    
        KEDF<CartesianOOPGrid> *tf = factory->constructKEDF(dat->grid,configTF);
        KEDF<CartesianOOPGrid> *vW = factory->constructKEDF(dat->grid,configvW);
        KEDF<CartesianOOPGrid> *wgc = factory->constructFourierKEDF(dat->grid,configWGC);
    
        shared_ptr<KEDF<CartesianOOPGrid> > tfSh(tf);
        shared_ptr<KEDF<CartesianOOPGrid> > vWSh(vW);
        shared_ptr<KEDF<CartesianOOPGrid> > wgcSh(wgc);
    
        weights->push_back(1.0);
        kedfs->push_back(tfSh);
        weights->push_back(1.0);
        kedfs->push_back(vWSh);
        weights->push_back(1.0);
        kedfs->push_back(wgcSh);
    
        dat->kedf = new MultiTermKEDF<CartesianOOPGrid>(dat->grid, move(kedfs), move(weights));
    
        delete factory;
    }
#ifdef LIBKEDF_OCL
    else if(dat->configuredGridType == OCL){
        KEDFFactory<CartesianOCLOOPGrid>* factory = new KEDFFactory<CartesianOCLOOPGrid>();
    
        ostringstream oss;
        oss << "1st-order-WangGovindCarter99: rho0= ";
        oss << (*rho0);
    
        const string configWGC = oss.str();
    
        unique_ptr<vector<double> > weights = make_unique<vector<double> >();
        unique_ptr<vector<shared_ptr<KEDF<CartesianOCLOOPGrid> > > > kedfs = make_unique<vector<shared_ptr<KEDF<CartesianOCLOOPGrid> > > >();
    
        KEDF<CartesianOCLOOPGrid> *tf = factory->constructKEDF(dat->gridOCL,configTF);
        KEDF<CartesianOCLOOPGrid> *vW = factory->constructKEDF(dat->gridOCL,configvW);
        KEDF<CartesianOCLOOPGrid> *wgc = factory->constructFourierKEDF(dat->gridOCL,configWGC);
    
        shared_ptr<KEDF<CartesianOCLOOPGrid> > tfSh(tf);
        shared_ptr<KEDF<CartesianOCLOOPGrid> > vWSh(vW);
        shared_ptr<KEDF<CartesianOCLOOPGrid> > wgcSh(wgc);
    
        weights->push_back(1.0);
        kedfs->push_back(tfSh);
        weights->push_back(1.0);
        kedfs->push_back(vWSh);
        weights->push_back(1.0);
        kedfs->push_back(wgcSh);
    
        dat->kedfOCL = new MultiTermKEDF<CartesianOCLOOPGrid>(dat->gridOCL, move(kedfs), move(weights));
    
        delete factory;
    }
#endif
    else{
        cerr << "Unknown case in libkedf_initialize_wgc1st_" << dat->configuredGridType << endl;
        throw runtime_error("Unknown case in libkedf_initialize_wgc1st_");
    }
}

void libkedf_initialize_wgc1st_custom_(struct libkedf_data *dat, const double *rho0, const double *alpha, const double *beta, const double *gamma){
    
    const string configTF = "ThomasFermi";
    const string configvW = "vonWeizsaecker";

    if(dat->configuredGridType == SMP){
        KEDFFactory<CartesianOOPGrid>* factory = new KEDFFactory<CartesianOOPGrid>();
    
        unique_ptr<vector<double> > weights = make_unique<vector<double> >();
        unique_ptr<vector<shared_ptr<KEDF<CartesianOOPGrid> > > > kedfs = make_unique<vector<shared_ptr<KEDF<CartesianOOPGrid> > > >();
    
        KEDF<CartesianOOPGrid> *tf = factory->constructKEDF(dat->grid,configTF);
        KEDF<CartesianOOPGrid> *vW = factory->constructKEDF(dat->grid,configvW);
        
        NumericalWangGovindCarterKernel* kernel = new NumericalWangGovindCarterKernel(*alpha, *beta, *gamma, *rho0);

        unique_ptr<CartesianOOPGrid> work = dat->grid->emptyDuplicate();
        cx_cube* recGrid = work->getReciprocalGrid();
        const size_t rows = recGrid->n_rows;
        const size_t cols = recGrid->n_cols;
        const size_t slices = recGrid->n_slices;

        cube* kernelCube0th = MemoryFunctions::allocateScratchCube(rows, cols, slices);
        cube* kernelCube1st = MemoryFunctions::allocateScratchCube(rows, cols, slices);
        const cube* gNorms = work->getGNorms();
        kernel->fillWGCKernel(kernelCube0th, kernelCube1st, gNorms);

        delete kernel;

        KEDF<CartesianOOPGrid> *wgc = new TayloredWangGovindCarter<CartesianOOPGrid>(work.get(), *alpha, *beta, *gamma, *rho0, kernelCube0th, kernelCube1st);
    
        shared_ptr<KEDF<CartesianOOPGrid> > tfSh(tf);
        shared_ptr<KEDF<CartesianOOPGrid> > vWSh(vW);
        shared_ptr<KEDF<CartesianOOPGrid> > wgcSh(wgc);
    
        weights->push_back(1.0);
        kedfs->push_back(tfSh);
        weights->push_back(1.0);
        kedfs->push_back(vWSh);
        weights->push_back(1.0);
        kedfs->push_back(wgcSh);
    
        dat->kedf = new MultiTermKEDF<CartesianOOPGrid>(dat->grid, move(kedfs), move(weights));
    
        delete factory;
    }
#ifdef LIBKEDF_OCL
    else if(dat->configuredGridType == OCL){
        KEDFFactory<CartesianOCLOOPGrid>* factory = new KEDFFactory<CartesianOCLOOPGrid>();
    
        unique_ptr<vector<double> > weights = make_unique<vector<double> >();
        unique_ptr<vector<shared_ptr<KEDF<CartesianOCLOOPGrid> > > > kedfs = make_unique<vector<shared_ptr<KEDF<CartesianOCLOOPGrid> > > >();
    
        KEDF<CartesianOCLOOPGrid> *tf = factory->constructKEDF(dat->gridOCL,configTF);
        KEDF<CartesianOCLOOPGrid> *vW = factory->constructKEDF(dat->gridOCL,configvW);
        
        NumericalWangGovindCarterKernel* kernel = new NumericalWangGovindCarterKernel(*alpha, *beta, *gamma, *rho0);

        unique_ptr<CartesianOCLOOPGrid> work = dat->gridOCL->emptyDuplicate();
        cx_cube* recGrid = work->getReciprocalGrid();
        const size_t rows = recGrid->n_rows;
        const size_t cols = recGrid->n_cols;
        const size_t slices = recGrid->n_slices;

        cube* kernelCube0th = MemoryFunctions::allocateScratchCube(rows, cols, slices);
        cube* kernelCube1st = MemoryFunctions::allocateScratchCube(rows, cols, slices);
        const cube* gNorms = work->getGNorms();
        kernel->fillWGCKernel(kernelCube0th, kernelCube1st, gNorms);

        delete kernel;

        KEDF<CartesianOCLOOPGrid> *wgc = new TayloredWangGovindCarter<CartesianOCLOOPGrid>(work.get(), *alpha, *beta, *gamma, *rho0, kernelCube0th, kernelCube1st);
    
        shared_ptr<KEDF<CartesianOCLOOPGrid> > tfSh(tf);
        shared_ptr<KEDF<CartesianOCLOOPGrid> > vWSh(vW);
        shared_ptr<KEDF<CartesianOCLOOPGrid> > wgcSh(wgc);
    
        weights->push_back(1.0);
        kedfs->push_back(tfSh);
        weights->push_back(1.0);
        kedfs->push_back(vWSh);
        weights->push_back(1.0);
        kedfs->push_back(wgcSh);
    
        dat->kedfOCL = new MultiTermKEDF<CartesianOCLOOPGrid>(dat->gridOCL, move(kedfs), move(weights));
    
        delete factory;
    }
#endif
    else{
        cerr << "Unknown case in libkedf_initialize_wgc1st_custom_" << dat->configuredGridType << endl;
        throw runtime_error("Unknown case in libkedf_initialize_wgc1st_custom_");
    }
}

void libkedf_initialize_wgc2nd_(struct libkedf_data *dat, const double *rho0){

    const string configTF = "ThomasFermi";
    const string configvW = "vonWeizsaecker";
    
    if(dat->configuredGridType == SMP){
        KEDFFactory<CartesianOOPGrid>* factory = new KEDFFactory<CartesianOOPGrid>();
    
        ostringstream oss;
        oss << "2nd-order-WangGovindCarter99: rho0= ";
        oss << (*rho0);
    
        const string configWGC = oss.str();
    
        unique_ptr<vector<double> > weights = make_unique<vector<double> >();
        unique_ptr<vector<shared_ptr<KEDF<CartesianOOPGrid> > > > kedfs = make_unique<vector<shared_ptr<KEDF<CartesianOOPGrid> > > >();
    
        KEDF<CartesianOOPGrid> *tf = factory->constructKEDF(dat->grid,configTF);
        KEDF<CartesianOOPGrid> *vW = factory->constructKEDF(dat->grid,configvW);
        KEDF<CartesianOOPGrid> *wgc = factory->constructFourierKEDF(dat->grid,configWGC);
    
        shared_ptr<KEDF<CartesianOOPGrid> > tfSh(tf);
        shared_ptr<KEDF<CartesianOOPGrid> > vWSh(vW);
        shared_ptr<KEDF<CartesianOOPGrid> > wgcSh(wgc);
    
        weights->push_back(1.0);
        kedfs->push_back(tfSh);
        weights->push_back(1.0);
        kedfs->push_back(vWSh);
        weights->push_back(1.0);
        kedfs->push_back(wgcSh);
    
        dat->kedf = new MultiTermKEDF<CartesianOOPGrid>(dat->grid, move(kedfs), move(weights));
    
        delete factory;
    }
#ifdef LIBKEDF_OCL
    else if(dat->configuredGridType == OCL){
        KEDFFactory<CartesianOCLOOPGrid>* factory = new KEDFFactory<CartesianOCLOOPGrid>();
    
        ostringstream oss;
        oss << "2nd-order-WangGovindCarter99: rho0= ";
        oss << (*rho0);
    
        const string configWGC = oss.str();
    
        unique_ptr<vector<double> > weights = make_unique<vector<double> >();
        unique_ptr<vector<shared_ptr<KEDF<CartesianOCLOOPGrid> > > > kedfs = make_unique<vector<shared_ptr<KEDF<CartesianOCLOOPGrid> > > >();
    
        KEDF<CartesianOCLOOPGrid> *tf = factory->constructKEDF(dat->gridOCL,configTF);
        KEDF<CartesianOCLOOPGrid> *vW = factory->constructKEDF(dat->gridOCL,configvW);
        KEDF<CartesianOCLOOPGrid> *wgc = factory->constructFourierKEDF(dat->gridOCL,configWGC);
    
        shared_ptr<KEDF<CartesianOCLOOPGrid> > tfSh(tf);
        shared_ptr<KEDF<CartesianOCLOOPGrid> > vWSh(vW);
        shared_ptr<KEDF<CartesianOCLOOPGrid> > wgcSh(wgc);
    
        weights->push_back(1.0);
        kedfs->push_back(tfSh);
        weights->push_back(1.0);
        kedfs->push_back(vWSh);
        weights->push_back(1.0);
        kedfs->push_back(wgcSh);
    
        dat->kedfOCL = new MultiTermKEDF<CartesianOCLOOPGrid>(dat->gridOCL, move(kedfs), move(weights));
    
        delete factory;
    }
#endif
    else{
        cerr << "Unknown case in libkedf_initialize_wgc2nd_" << dat->configuredGridType << endl;
        throw runtime_error("Unknown case in libkedf_initialize_wgc2nd_");
    }
}

void libkedf_initialize_wgc2nd_custom_(struct libkedf_data *dat, const double *rho0, const double *alpha, const double *beta, const double *gamma){
    
    const string configTF = "ThomasFermi";
    const string configvW = "vonWeizsaecker";
    
    if(dat->configuredGridType == SMP){
        KEDFFactory<CartesianOOPGrid>* factory = new KEDFFactory<CartesianOOPGrid>();
    
        unique_ptr<vector<double> > weights = make_unique<vector<double> >();
        unique_ptr<vector<shared_ptr<KEDF<CartesianOOPGrid> > > > kedfs = make_unique<vector<shared_ptr<KEDF<CartesianOOPGrid> > > >();
    
        KEDF<CartesianOOPGrid> *tf = factory->constructKEDF(dat->grid,configTF);
        KEDF<CartesianOOPGrid> *vW = factory->constructKEDF(dat->grid,configvW);
        
        NumericalWangGovindCarterKernel* kernel = new NumericalWangGovindCarterKernel(*alpha, *beta, *gamma, *rho0);

        unique_ptr<CartesianOOPGrid> work = dat->grid->emptyDuplicate();
        cx_cube* recGrid = work->getReciprocalGrid();
        const size_t rows = recGrid->n_rows;
        const size_t cols = recGrid->n_cols;
        const size_t slices = recGrid->n_slices;

        cube* kernelCube0th = MemoryFunctions::allocateScratchCube(rows, cols, slices);
        cube* kernelCube1st = MemoryFunctions::allocateScratchCube(rows, cols, slices);
        cube* kernelCube2nd = MemoryFunctions::allocateScratchCube(rows, cols, slices);
        cube* kernelCube3rd = MemoryFunctions::allocateScratchCube(rows, cols, slices);
        const cube* gNorms = work->getGNorms();
        kernel->fillWGCKernel(kernelCube0th, kernelCube1st, kernelCube2nd, kernelCube3rd, gNorms);

        delete kernel;
        
        KEDF<CartesianOOPGrid> *wgc = new TayloredWangGovindCarter<CartesianOOPGrid>(work.get(), *alpha, *beta, *gamma, *rho0, kernelCube0th, kernelCube1st,
                    kernelCube2nd, kernelCube3rd);
    
        shared_ptr<KEDF<CartesianOOPGrid> > tfSh(tf);
        shared_ptr<KEDF<CartesianOOPGrid> > vWSh(vW);
        shared_ptr<KEDF<CartesianOOPGrid> > wgcSh(wgc);
    
        weights->push_back(1.0);
        kedfs->push_back(tfSh);
        weights->push_back(1.0);
        kedfs->push_back(vWSh);
        weights->push_back(1.0);
        kedfs->push_back(wgcSh);
    
        dat->kedf = new MultiTermKEDF<CartesianOOPGrid>(dat->grid, move(kedfs), move(weights));
    
        delete factory;
    }
#ifdef LIBKEDF_OCL
    else if(dat->configuredGridType == OCL){
        KEDFFactory<CartesianOCLOOPGrid>* factory = new KEDFFactory<CartesianOCLOOPGrid>();
    
        unique_ptr<vector<double> > weights = make_unique<vector<double> >();
        unique_ptr<vector<shared_ptr<KEDF<CartesianOCLOOPGrid> > > > kedfs = make_unique<vector<shared_ptr<KEDF<CartesianOCLOOPGrid> > > >();
    
        KEDF<CartesianOCLOOPGrid> *tf = factory->constructKEDF(dat->gridOCL,configTF);
        KEDF<CartesianOCLOOPGrid> *vW = factory->constructKEDF(dat->gridOCL,configvW);
        
        NumericalWangGovindCarterKernel* kernel = new NumericalWangGovindCarterKernel(*alpha, *beta, *gamma, *rho0);

        unique_ptr<CartesianOCLOOPGrid> work = dat->gridOCL->emptyDuplicate();
        cx_cube* recGrid = work->getReciprocalGrid();
        const size_t rows = recGrid->n_rows;
        const size_t cols = recGrid->n_cols;
        const size_t slices = recGrid->n_slices;

        cube* kernelCube0th = MemoryFunctions::allocateScratchCube(rows, cols, slices);
        cube* kernelCube1st = MemoryFunctions::allocateScratchCube(rows, cols, slices);
        cube* kernelCube2nd = MemoryFunctions::allocateScratchCube(rows, cols, slices);
        cube* kernelCube3rd = MemoryFunctions::allocateScratchCube(rows, cols, slices);
        const cube* gNorms = work->getGNorms();
        kernel->fillWGCKernel(kernelCube0th, kernelCube1st, kernelCube2nd, kernelCube3rd, gNorms);

        delete kernel;
        
        KEDF<CartesianOCLOOPGrid> *wgc = new TayloredWangGovindCarter<CartesianOCLOOPGrid>(work.get(), *alpha, *beta, *gamma, *rho0, kernelCube0th, kernelCube1st,
                    kernelCube2nd, kernelCube3rd);
    
        shared_ptr<KEDF<CartesianOCLOOPGrid> > tfSh(tf);
        shared_ptr<KEDF<CartesianOCLOOPGrid> > vWSh(vW);
        shared_ptr<KEDF<CartesianOCLOOPGrid> > wgcSh(wgc);
    
        weights->push_back(1.0);
        kedfs->push_back(tfSh);
        weights->push_back(1.0);
        kedfs->push_back(vWSh);
        weights->push_back(1.0);
        kedfs->push_back(wgcSh);
    
        dat->kedfOCL = new MultiTermKEDF<CartesianOCLOOPGrid>(dat->gridOCL, move(kedfs), move(weights));
    
        delete factory;
    }
#endif
    else{
        cerr << "Unknown case in libkedf_initialize_wgc2nd_custom_" << endl;
    }
}
    
void libkedf_initialize_hc_(struct libkedf_data *dat, const double *rho0, const double *lambda){
    
    const string configTF = "ThomasFermi";
    const string configvW = "vonWeizsaecker";
    
    if(dat->configuredGridType == SMP){
        KEDFFactory<CartesianOOPGrid>* factory = new KEDFFactory<CartesianOOPGrid>();
        
        unique_ptr<vector<double> > weights = make_unique<vector<double> >();
        unique_ptr<vector<shared_ptr<KEDF<CartesianOOPGrid> > > > kedfs = make_unique<vector<shared_ptr<KEDF<CartesianOOPGrid> > > >();
    
        KEDF<CartesianOOPGrid> *tf = factory->constructKEDF(dat->grid,configTF);
        KEDF<CartesianOOPGrid> *vW = factory->constructKEDF(dat->grid,configvW);
    
        const double alpha = 2.0166666666666666;
        const double beta = 0.65;
        const double c = 8*3*M_PI*M_PI;
        const double refRatio = 1.15;
            
        // get the ODE part of the kernel setup
        const double wInf = -8.0/3.0/((5.0-3.0*beta)*beta);
    
        HuangCarterODE* ode = new HuangCarterODE(beta);

        IntKernelODE* odeInt =  new IntKernelODE(true); // XXX stats for the time being
        unique_ptr<mat> fullW = odeInt->makeFirstOrderKernel(ode,wInf);
    
        const size_t numEta = fullW->n_rows;
        const double etaStep = 0.001; // XXX hard-coded, sorry
    
        unique_ptr<vec> w = make_unique<vec>(numEta);
        unique_ptr<vec> w1 = make_unique<vec>(numEta);
        for(size_t i = 0; i < numEta; ++i){
            w->at(i) = fullW->at(i,0);
            w1->at(i) = fullW->at(i,1);
        }
    
        // these are sensible (but slow) defaults for the time being
        const double cutoffDens = 0.0;
        const bool trashEmptyBins = false;
        const size_t cutoffEmptyBin = 0;
    
        delete ode;
        delete odeInt;
            
        KEDF<CartesianOOPGrid> *hc = new HuangCarter<CartesianOOPGrid>(dat->grid, numEta, etaStep, move(w), move(w1),
            alpha, beta, *lambda, *rho0, c, refRatio, cutoffDens, trashEmptyBins, cutoffEmptyBin);
    
        shared_ptr<KEDF<CartesianOOPGrid> > tfSh(tf);
        shared_ptr<KEDF<CartesianOOPGrid> > vWSh(vW);
        shared_ptr<KEDF<CartesianOOPGrid> > hcSh(hc);
    
        weights->push_back(1.0);
        kedfs->push_back(tfSh);
        weights->push_back(1.0);
        kedfs->push_back(vWSh);
        weights->push_back(1.0);
        kedfs->push_back(hcSh);
    
        dat->kedf = new MultiTermKEDF<CartesianOOPGrid>(dat->grid, move(kedfs), move(weights));
    
        delete factory;
    }
#ifdef LIBKEDF_OCL
    else if(dat->configuredGridType == OCL){
        KEDFFactory<CartesianOCLOOPGrid>* factory = new KEDFFactory<CartesianOCLOOPGrid>();
        
        unique_ptr<vector<double> > weights = make_unique<vector<double> >();
        unique_ptr<vector<shared_ptr<KEDF<CartesianOCLOOPGrid> > > > kedfs = make_unique<vector<shared_ptr<KEDF<CartesianOCLOOPGrid> > > >();
    
        KEDF<CartesianOCLOOPGrid> *tf = factory->constructKEDF(dat->gridOCL,configTF);
        KEDF<CartesianOCLOOPGrid> *vW = factory->constructKEDF(dat->gridOCL,configvW);
    
        const double alpha = 2.0166666666666666;
        const double beta = 0.65;
        const double c = 8*3*M_PI*M_PI;
        const double refRatio = 1.15;
            
        // get the ODE part of the kernel setup
        const double wInf = -8.0/3.0/((5.0-3.0*beta)*beta);
    
        HuangCarterODE* ode = new HuangCarterODE(beta);

        IntKernelODE* odeInt =  new IntKernelODE(true); // XXX stats for the time being
        unique_ptr<mat> fullW = odeInt->makeFirstOrderKernel(ode,wInf);
    
        const size_t numEta = fullW->n_rows;
        const double etaStep = 0.001; // XXX hard-coded, sorry
    
        unique_ptr<vec> w = make_unique<vec>(numEta);
        unique_ptr<vec> w1 = make_unique<vec>(numEta);
        for(size_t i = 0; i < numEta; ++i){
            w->at(i) = fullW->at(i,0);
            w1->at(i) = fullW->at(i,1);
        }
    
        // these are sensible (but slow) defaults for the time being
        const double cutoffDens = 0.0;
        const bool trashEmptyBins = false;
        const size_t cutoffEmptyBin = 0;
    
        delete ode;
        delete odeInt;
            
        KEDF<CartesianOCLOOPGrid> *hc = new HuangCarter<CartesianOCLOOPGrid>(dat->gridOCL, numEta, etaStep, move(w), move(w1),
            alpha, beta, *lambda, *rho0, c, refRatio, cutoffDens, trashEmptyBins, cutoffEmptyBin);
    
        shared_ptr<KEDF<CartesianOCLOOPGrid> > tfSh(tf);
        shared_ptr<KEDF<CartesianOCLOOPGrid> > vWSh(vW);
        shared_ptr<KEDF<CartesianOCLOOPGrid> > hcSh(hc);
    
        weights->push_back(1.0);
        kedfs->push_back(tfSh);
        weights->push_back(1.0);
        kedfs->push_back(vWSh);
        weights->push_back(1.0);
        kedfs->push_back(hcSh);
    
        dat->kedfOCL = new MultiTermKEDF<CartesianOCLOOPGrid>(dat->gridOCL, move(kedfs), move(weights));
    
        delete factory;
    }
#endif
    else{
        cerr << "Unknown case in libkedf_initialize_hc_" << endl;
    }
}

void libkedf_initialize_hc_custom_(struct libkedf_data *dat, const double *rho0, const double *lambda, const double *alpha, const double *beta, const double *refRatio){
    
    const string configTF = "ThomasFermi";
    const string configvW = "vonWeizsaecker";
    
    if(dat->configuredGridType == SMP){
        KEDFFactory<CartesianOOPGrid>* factory = new KEDFFactory<CartesianOOPGrid>();
        
        unique_ptr<vector<double> > weights = make_unique<vector<double> >();
        unique_ptr<vector<shared_ptr<KEDF<CartesianOOPGrid> > > > kedfs = make_unique<vector<shared_ptr<KEDF<CartesianOOPGrid> > > >();
    
        KEDF<CartesianOOPGrid> *tf = factory->constructKEDF(dat->grid,configTF);
        KEDF<CartesianOOPGrid> *vW = factory->constructKEDF(dat->grid,configvW);
    
        const double c = 8*3*M_PI*M_PI;
            
        // get the ODE part of the kernel setup
        const double wInf = -8.0/3.0/((5.0-3.0*(*beta))*(*beta));
    
        HuangCarterODE* ode = new HuangCarterODE(*beta);

        IntKernelODE* odeInt =  new IntKernelODE(true); // XXX stats for the time being
        unique_ptr<mat> fullW = odeInt->makeFirstOrderKernel(ode,wInf);
    
        const size_t numEta = fullW->n_rows;
        const double etaStep = 0.001; // XXX hard-coded, sorry
    
        unique_ptr<vec> w = make_unique<vec>(numEta);
        unique_ptr<vec> w1 = make_unique<vec>(numEta);
        for(size_t i = 0; i < numEta; ++i){
            w->at(i) = fullW->at(i,0);
            w1->at(i) = fullW->at(i,1);
        }
    
        // these are sensible (but slow) defaults for the time being
        const double cutoffDens = 0.0;
        const bool trashEmptyBins = false;
        const size_t cutoffEmptyBin = 0;
    
        delete ode;
        delete odeInt;
            
        KEDF<CartesianOOPGrid> *hc = new HuangCarter<CartesianOOPGrid>(dat->grid, numEta, etaStep, move(w), move(w1),
            *alpha, *beta, *lambda, *rho0, c, *refRatio, cutoffDens, trashEmptyBins, cutoffEmptyBin);
    
        shared_ptr<KEDF<CartesianOOPGrid> > tfSh(tf);
        shared_ptr<KEDF<CartesianOOPGrid> > vWSh(vW);
        shared_ptr<KEDF<CartesianOOPGrid> > hcSh(hc);
    
        weights->push_back(1.0);
        kedfs->push_back(tfSh);
        weights->push_back(1.0);
        kedfs->push_back(vWSh);
        weights->push_back(1.0);
        kedfs->push_back(hcSh);
    
        dat->kedf = new MultiTermKEDF<CartesianOOPGrid>(dat->grid, move(kedfs), move(weights));
    
        delete factory;
    }
#ifdef LIBKEDF_OCL
    else if(dat->configuredGridType == OCL){
        KEDFFactory<CartesianOCLOOPGrid>* factory = new KEDFFactory<CartesianOCLOOPGrid>();
        
        unique_ptr<vector<double> > weights = make_unique<vector<double> >();
        unique_ptr<vector<shared_ptr<KEDF<CartesianOCLOOPGrid> > > > kedfs = make_unique<vector<shared_ptr<KEDF<CartesianOCLOOPGrid> > > >();
    
        KEDF<CartesianOCLOOPGrid> *tf = factory->constructKEDF(dat->gridOCL,configTF);
        KEDF<CartesianOCLOOPGrid> *vW = factory->constructKEDF(dat->gridOCL,configvW);
    
        const double c = 8*3*M_PI*M_PI;
            
        // get the ODE part of the kernel setup
        const double wInf = -8.0/3.0/((5.0-3.0*(*beta))*(*beta));
    
        HuangCarterODE* ode = new HuangCarterODE(*beta);

        IntKernelODE* odeInt =  new IntKernelODE(true); // XXX stats for the time being
        unique_ptr<mat> fullW = odeInt->makeFirstOrderKernel(ode,wInf);
    
        const size_t numEta = fullW->n_rows;
        const double etaStep = 0.001; // XXX hard-coded, sorry
    
        unique_ptr<vec> w = make_unique<vec>(numEta);
        unique_ptr<vec> w1 = make_unique<vec>(numEta);
        for(size_t i = 0; i < numEta; ++i){
            w->at(i) = fullW->at(i,0);
            w1->at(i) = fullW->at(i,1);
        }
    
        // these are sensible (but slow) defaults for the time being
        const double cutoffDens = 0.0;
        const bool trashEmptyBins = false;
        const size_t cutoffEmptyBin = 0;
    
        delete ode;
        delete odeInt;
            
        KEDF<CartesianOCLOOPGrid> *hc = new HuangCarter<CartesianOCLOOPGrid>(dat->gridOCL, numEta, etaStep, move(w), move(w1),
            *alpha, *beta, *lambda, *rho0, c, *refRatio, cutoffDens, trashEmptyBins, cutoffEmptyBin);
    
        shared_ptr<KEDF<CartesianOCLOOPGrid> > tfSh(tf);
        shared_ptr<KEDF<CartesianOCLOOPGrid> > vWSh(vW);
        shared_ptr<KEDF<CartesianOCLOOPGrid> > hcSh(hc);
    
        weights->push_back(1.0);
        kedfs->push_back(tfSh);
        weights->push_back(1.0);
        kedfs->push_back(vWSh);
        weights->push_back(1.0);
        kedfs->push_back(hcSh);
    
        dat->kedfOCL = new MultiTermKEDF<CartesianOCLOOPGrid>(dat->gridOCL, move(kedfs), move(weights));
    
        delete factory;
    }
#endif
    else{
        cerr << "Unknown case in libkedf_initialize_hc_custom_" << endl;
    }
}
    
void libkedf_energy_(struct libkedf_data *dat, const double *density, double *energy){
    
    if(dat->configuredGridType == SMP){
        dat->grid->updateRealGrid(density);
        double e = dat->kedf->calcEnergy(*(dat->grid));
        energy[0] = e;
    }
#ifdef LIBKEDF_OCL
    else if(dat->configuredGridType == OCL){
        dat->gridOCL->updateRealGrid(density);
        double e = dat->kedfOCL->calcEnergy(*(dat->gridOCL));
        energy[0] = e;
    }
#endif
    else{
        cerr << "Unknown case in libkedf_energy_" << endl;
    }
}
    
void libkedf_potential_(struct libkedf_data *dat, const double *density, double *potential, double *energy){
    
    if(dat->configuredGridType == SMP){
        dat->grid->updateRealGrid(density);
        if(!dat->potential){
            dat->potential = dat->grid->emptyDuplicate().release();
        }
        double e = dat->kedf->calcPotential(*(dat->grid), *(dat->potential));
    
        dat->potential->getRealGridData(potential);
    
        energy[0] = e;
    }
#ifdef LIBKEDF_OCL
    else if(dat->configuredGridType == OCL){
        dat->gridOCL->updateRealGrid(density);
        if(!dat->potentialOCL){
            dat->potentialOCL = dat->gridOCL->emptyDuplicate().release();
        }
        double e = dat->kedfOCL->calcPotential(*(dat->gridOCL), *(dat->potentialOCL));
    
        dat->potentialOCL->getRealGridData(potential);
    
        energy[0] = e;
    }
#endif
    else{
        cerr << "Unknown case in libkedf_potential_" << endl;
    }
}

void libkedf_cleanup_(struct libkedf_data *dat){
    
    if(!dat){return;}
    
    if(dat->grid){
        delete dat->grid;
    }
    if(dat->kedf){
        delete dat->kedf;
    }
    if(dat->potential){
        delete dat->potential;
    }
    
#ifdef LIBKEDF_OCL
    if(dat->gridOCL){
        delete dat->gridOCL;
    }
    if(dat->kedfOCL){
        delete dat->kedfOCL;
    }
    if(dat->potentialOCL){
        delete dat->potentialOCL;
    }
#endif
    
    // XXX otherwise segfault
    //free(dat);
}

#ifdef __cplusplus
}
#endif
