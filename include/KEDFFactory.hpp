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

#ifndef KEDFFACTORY_HPP
#define	KEDFFACTORY_HPP

#include <cmath>
#include <string>
#include "CartesianOOPGrid.hpp"
#ifdef LIBKEDF_OCL
#include "CartesianOCLOOPGrid.hpp"
#endif
#include "HuangCarter.hpp"
#include "ThomasFermi.hpp"
#include "VonWeizsaecker.hpp"
#include "WangGovindCarter.hpp"
#include "WangTeter.hpp"
#include "KEDF.hpp"
#include "HelperFunctions.hpp"
using namespace std;

template<typename GridType>
class KEDFFactory {
public:
    
    KEDFFactory();
    ~KEDFFactory();
    
    KEDF<GridType>* constructKEDF(GridType* grid, const string config);
    
    KEDF<GridType>* constructFourierKEDF(GridType* grid, const string config);
};

template<>
class KEDFFactory <CartesianOOPGrid> {
public:
    
    KEDFFactory(){}
    ~KEDFFactory(){}
    
    KEDF<CartesianOOPGrid>* constructKEDF(CartesianOOPGrid* grid, const string config) {
        
        if(config.compare("ThomasFermi") == 0){
            return (new ThomasFermi<CartesianOOPGrid>(grid));
        } else if(config.compare("vonWeizsaecker") == 0){
            return (new VonWeizsaecker<CartesianOOPGrid>(grid));
        } else {
            throw runtime_error("Unknown KEDF definition " + config);
        }
    
        throw runtime_error("Missing return for KEDF definition " + config);
    }
    
    KEDF<CartesianOOPGrid>* constructFourierKEDF(CartesianOOPGrid* grid, const string config) {
        
        if (config.compare(0,10,"WangTeter:") == 0) {

            const double alpha = 5.0 / 6.0;
            const double beta = 5.0 / 6.0;
            
            double rho0;
            string tmp;
            stringstream ss(config);
            ss >> tmp; // get rid of the KEDF string
            ss >> tmp; // get rid of the prefix (ugly...))
            ss >> rho0; // get rho0
            
            const double lambdaTF = 1.0;
            const double muVW = 1.0;
            const double ft = 5.0 / 3.0;

            WangTeterKernel* kernel = new WangTeterKernel(alpha, beta, rho0, lambdaTF, muVW, ft);

            unique_ptr<CartesianOOPGrid> work = grid->emptyDuplicate();
            cx_cube* recGrid = work->getReciprocalGrid();
            const size_t rows = recGrid->n_rows;
            const size_t cols = recGrid->n_cols;
            const size_t slices = recGrid->n_slices;

            cube* kernelCube = MemoryFunctions::allocateScratchCube(rows, cols, slices);
            const cube* gNorms = work->getGNorms();
            kernel->fillWTKernelReciprocal(kernelCube, gNorms);

            delete kernel;

            return (new WangTeter<CartesianOOPGrid>(grid, alpha, beta, kernelCube));
        } else if (config.compare(0,17,"SmargiassiMadden:") == 0) {

            const double alpha = 0.5;
            const double beta = 0.5;
            
            double rho0;
            string tmp;
            stringstream ss(config);
            ss >> tmp; // get rid of the KEDF string
            ss >> tmp; // get rid of the prefix (ugly...))
            ss >> rho0; // get rho0
            
            const double lambdaTF = 1.0;
            const double muVW = 1.0;
            const double ft = 5.0 / 3.0;

            WangTeterKernel* kernel = new WangTeterKernel(alpha, beta, rho0, lambdaTF, muVW, ft);

            unique_ptr<CartesianOOPGrid> work = grid->emptyDuplicate();
            cx_cube* recGrid = work->getReciprocalGrid();
            const size_t rows = recGrid->n_rows;
            const size_t cols = recGrid->n_cols;
            const size_t slices = recGrid->n_slices;

            cube* kernelCube = MemoryFunctions::allocateScratchCube(rows, cols, slices);
            const cube* gNorms = work->getGNorms();
            kernel->fillWTKernelReciprocal(kernelCube, gNorms);

            delete kernel;

            return (new WangTeter<CartesianOOPGrid>(grid, alpha, beta, kernelCube));
        } else if (config.compare(0,29,"1st-order-WangGovindCarter99:") == 0){
            
            const double alpha = 1.20601132958330;
            const double beta = 0.460655337083368;
            const double gamma = 2.7;
            
            double rho0;
            string tmp;
            stringstream ss(config);
            ss >> tmp; // get rid of the KEDF string
            ss >> tmp; // get rid of the prefix (ugly...))
            ss >> rho0; // get rho0
            
            NumericalWangGovindCarterKernel* kernel = new NumericalWangGovindCarterKernel(alpha, beta, gamma, rho0);

            unique_ptr<CartesianOOPGrid> work = grid->emptyDuplicate();
            cx_cube* recGrid = work->getReciprocalGrid();
            const size_t rows = recGrid->n_rows;
            const size_t cols = recGrid->n_cols;
            const size_t slices = recGrid->n_slices;

            cube* kernelCube0th = MemoryFunctions::allocateScratchCube(rows, cols, slices);
            cube* kernelCube1st = MemoryFunctions::allocateScratchCube(rows, cols, slices);
            const cube* gNorms = work->getGNorms();
            kernel->fillWGCKernel(kernelCube0th, kernelCube1st, gNorms);

            delete kernel;
            

            return (new TayloredWangGovindCarter<CartesianOOPGrid>(grid, alpha, beta, gamma, rho0, kernelCube0th, kernelCube1st));            
            
        } else if (config.compare(0,29,"2nd-order-WangGovindCarter99:") == 0){
            
            const double alpha = 1.20601132958330;
            const double beta = 0.460655337083368;
            const double gamma = 2.7;
            
            double rho0;
            string tmp;
            stringstream ss(config);
            ss >> tmp; // get rid of the KEDF string
            ss >> tmp; // get rid of the prefix (ugly...))
            ss >> rho0; // get rho0
            
            NumericalWangGovindCarterKernel* kernel = new NumericalWangGovindCarterKernel(alpha, beta, gamma, rho0);

            unique_ptr<CartesianOOPGrid> work = grid->emptyDuplicate();
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
            
            return (new TayloredWangGovindCarter<CartesianOOPGrid>(grid, alpha, beta, gamma, rho0, kernelCube0th, kernelCube1st,
                    kernelCube2nd, kernelCube3rd));
            
        } else if (config.compare(0,12,"HuangCarter:") == 0){
            
            const double alpha = 2.0166666666666666;
            const double beta = 0.65;
            const double lambda = 0.01;
            const double c = 8*3*M_PI*M_PI;
            const double refRatio = 1.15;
            
            double rho0;
            string tmp;
            stringstream ss(config);
            ss >> tmp; // get rid of the KEDF string
            ss >> tmp; // get rid of the prefix (ugly...))
            ss >> rho0; // get rho0
            
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
            
            return (new HuangCarter<CartesianOOPGrid>(grid, numEta, etaStep, move(w), move(w1),
                    alpha, beta, lambda, rho0, c, refRatio, cutoffDens, trashEmptyBins, cutoffEmptyBin));
        } else {
            throw runtime_error("Unknown FourierKEDF definition " + config);
        }

        throw runtime_error("Missing return for FourierKEDF definition " + config);
    }
};

#ifdef LIBKEDF_OCL
template<>
class KEDFFactory <CartesianOCLOOPGrid> {
public:
    
    KEDFFactory(){}
    ~KEDFFactory(){}
    
    KEDF<CartesianOCLOOPGrid>* constructKEDF(CartesianOCLOOPGrid* grid, const string config){
        
        if(config.compare("ThomasFermi") == 0){
            return (new ThomasFermi<CartesianOCLOOPGrid>(grid));
        } else if(config.compare("vonWeizsaecker") == 0){
            grid->transferGNorms(); // we will need them a lot
            return (new VonWeizsaecker<CartesianOCLOOPGrid>(grid));
        } else {
            throw runtime_error("Unknown KEDF definition " + config);
        }
    
        throw runtime_error("Missing return for KEDF definition " + config);
    }
    
    KEDF<CartesianOCLOOPGrid>* constructFourierKEDF(CartesianOCLOOPGrid* grid, const string config) {
        
        if (config.compare(0,10,"WangTeter:") == 0) {

            const double alpha = 5.0 / 6.0;
            const double beta = 5.0 / 6.0;

            double rho0;
            string tmp;
            stringstream ss(config);
            ss >> tmp; // get rid of the KEDF string
            ss >> tmp; // get rid of the prefix (ugly...))
            ss >> rho0; // get rho0
            
            const double lambdaTF = 1.0;
            const double muVW = 1.0;
            const double ft = 5.0 / 3.0;

            WangTeterKernel* kernel = new WangTeterKernel(alpha, beta, rho0, lambdaTF, muVW, ft);

            unique_ptr<CartesianOCLOOPGrid> work = grid->emptyDuplicate();
            cx_cube* recGrid = work->getReciprocalGrid();
            const size_t rows = recGrid->n_rows;
            const size_t cols = recGrid->n_cols;
            const size_t slices = recGrid->n_slices;

            cube* kernelCube = MemoryFunctions::allocateScratchCube(rows, cols, slices);
            const cube* gNorms = work->getGNorms();
            kernel->fillWTKernelReciprocal(kernelCube, gNorms);

            delete kernel;

            return (new WangTeter<CartesianOCLOOPGrid>(grid, alpha, beta, kernelCube));
        } else if (config.compare(0,17,"SmargiassiMadden:") == 0) {

            const double alpha = 0.5;
            const double beta = 0.5;

            double rho0;
            string tmp;
            stringstream ss(config);
            ss >> tmp; // get rid of the KEDF string
            ss >> tmp; // get rid of the prefix (ugly...))
            ss >> rho0; // get rho0
            
            const double lambdaTF = 1.0;
            const double muVW = 1.0;
            const double ft = 5.0 / 3.0;

            WangTeterKernel* kernel = new WangTeterKernel(alpha, beta, rho0, lambdaTF, muVW, ft);

            unique_ptr<CartesianOCLOOPGrid> work = grid->emptyDuplicate();
            cx_cube* recGrid = work->getReciprocalGrid();
            const size_t rows = recGrid->n_rows;
            const size_t cols = recGrid->n_cols;
            const size_t slices = recGrid->n_slices;

            cube* kernelCube = MemoryFunctions::allocateScratchCube(rows, cols, slices);
            const cube* gNorms = work->getGNorms();
            kernel->fillWTKernelReciprocal(kernelCube, gNorms);

            delete kernel;

            return (new WangTeter<CartesianOCLOOPGrid>(grid, alpha, beta, kernelCube));
        } else if (config.compare(0,29,"1st-order-WangGovindCarter99:") == 0){
            
            const double alpha = 1.20601132958330;
            const double beta = 0.460655337083368;
            const double gamma = 2.7;
            
            double rho0;
            string tmp;
            stringstream ss(config);
            ss >> tmp; // get rid of the KEDF string
            ss >> tmp; // get rid of the prefix (ugly...))
            ss >> rho0; // get rho0
            
            NumericalWangGovindCarterKernel* kernel = new NumericalWangGovindCarterKernel(alpha, beta, gamma, rho0);

            unique_ptr<CartesianOCLOOPGrid> work = grid->emptyDuplicate();
            cx_cube* recGrid = work->getReciprocalGrid();
            const size_t rows = recGrid->n_rows;
            const size_t cols = recGrid->n_cols;
            const size_t slices = recGrid->n_slices;

            cube* kernelCube0th = MemoryFunctions::allocateScratchCube(rows, cols, slices);
            cube* kernelCube1st = MemoryFunctions::allocateScratchCube(rows, cols, slices);
            const cube* gNorms = work->getGNorms();
            kernel->fillWGCKernel(kernelCube0th, kernelCube1st, gNorms);

            delete kernel;
            

            return (new TayloredWangGovindCarter<CartesianOCLOOPGrid>(grid, alpha, beta, gamma, rho0, kernelCube0th, kernelCube1st));            
            
        } else if (config.compare(0,29,"2nd-order-WangGovindCarter99:") == 0){
            
            const double alpha = 1.20601132958330;
            const double beta = 0.460655337083368;
            const double gamma = 2.7;
            
            double rho0;
            string tmp;
            stringstream ss(config);
            ss >> tmp; // get rid of the KEDF string
            ss >> tmp; // get rid of the prefix (ugly...))
            ss >> rho0; // get rho0
            
            NumericalWangGovindCarterKernel* kernel = new NumericalWangGovindCarterKernel(alpha, beta, gamma, rho0);

            unique_ptr<CartesianOCLOOPGrid> work = grid->emptyDuplicate();
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
            
            return (new TayloredWangGovindCarter<CartesianOCLOOPGrid>(grid, alpha, beta, gamma, rho0, kernelCube0th, kernelCube1st,
                    kernelCube2nd, kernelCube3rd));
            
        } else if (config.compare(0,12,"HuangCarter:") == 0){
            
            const double alpha = 2.0166666666666666;
            const double beta = 0.65;
            const double lambda = 0.01;
            const double c = 8*3*M_PI*M_PI;
            const double refRatio = 1.15;
            
            double rho0;
            string tmp;
            stringstream ss(config);
            ss >> tmp; // get rid of the KEDF string
            ss >> tmp; // get rid of the prefix (ugly...))
            ss >> rho0; // get rho0
            
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
            
            return (new HuangCarter<CartesianOCLOOPGrid>(grid, numEta, etaStep, move(w), move(w1),
                    alpha, beta, lambda, rho0, c, refRatio, cutoffDens, trashEmptyBins, cutoffEmptyBin));
            
        } else {
            throw runtime_error("Unknown FourierKEDF definition " + config);
        }

        throw runtime_error("Missing return for FourierKEDF definition " + config);
    }
};
#endif



#endif	/* KEDFFACTORY_HPP */

