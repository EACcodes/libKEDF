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
#include <armadillo>
#ifdef _OPENMP
#include <fftw3.h>
#endif
#include "GridFactory.hpp"
#include "BasicGridComputer.hpp"
#include "CartesianOOPGrid.hpp"
#ifdef LIBKEDF_OCL
#include <clFFT.h>
#include "CartesianOCLOOPGrid.hpp"
#endif
#include <memory>
#include <stdio.h>
#include <time.h>
#include "Configuration.hpp"
#include "FourierGrid.hpp"
#include "FourierKEDF.hpp"
#include "GridFactory.hpp"
#include "GridFiller.hpp"
#include "KEDF.hpp"
#include "KEDFFactory.hpp"
#include "FourierGrid.hpp"
#include "StressTensor.hpp"
using namespace arma;
using namespace std;

template<class GridType>
void runKEDF(const Configuration* config, const KEDF<GridType>* kedf, const GridType* grid){
    
    cout << "Setup complete. Running KEDF " << kedf->getMethodDescription() << endl;
    cout << "Required citations:" << endl;
    const vector<string> citations = kedf->getCitations();
    for(size_t i = 0; i < citations.size(); ++i){
        cout << "" << (i+1) << ") " << citations[i] << endl;  
    }
    
    cout << endl;
    cout << "Working equations:" << endl; 
    const vector<string> equations = kedf->getWorkingEquations();
    for(size_t i = 0; i < equations.size(); ++i){
        cout << equations[i] << endl;
    }
    cout << endl;
    
    // do whatever operation was asked for and print results
    const int noIterations = config->getNoIterations();
    job_types_t jobType = config->getJobType();
    
    time_t timerStart;
    time_t timerEnd;
    
    time_t totalTimerStart;
    time_t totalTimerEnd;
    
    time(&totalTimerStart);
    
    for(int iter = 0; iter < noIterations; ++iter){
        
        time(&timerStart);
        
        switch(jobType){
            case ENERGY:
            {
                cout << "###########################################" << endl;
                const double en = kedf->calcEnergy(*grid);
                time(&timerEnd);
                cout << "Iteration " << iter << endl;
                cout << "Energy: " << en << endl;
            }
                break;
            case POTENTIAL:
            {
                unique_ptr<GridType> potential = grid->emptyDuplicate();
                cout << "###########################################" << endl;
                const double enGrad = kedf->calcPotential(*grid,*potential.get());
                time(&timerEnd);
                cout << "Iteration " << iter << endl;
                cout << "Energy: " << enGrad << endl;
                const arma::cube* pot = potential->readRealGrid();
                if(config->printVerbose()){
                    pot->print("Potential:");
                }
            }
                break;
            case STRESS:
                cout << "###########################################" << endl;
                unique_ptr<StressTensor> stress = kedf->calcStress(*grid);
                time(&timerEnd);
                cout << "Iteration " << iter << endl;
                const arma::mat* const tensor = stress->getTensor();
                tensor->print("Stress tensor:");
                break;
        }
        
        const double seconds = difftime(timerEnd,timerStart);
        printf ("KEDF evaluation took %.f seconds\n", seconds);
        
    }
    
    time(&totalTimerEnd);
    const double seconds = difftime(totalTimerEnd,totalTimerStart);
    printf ("All %d KEDF evaluations combined took %.f seconds\n", noIterations, seconds);    
}

int fillGrid(Grid* grid, const Configuration* config){
    
    try{
        const fillstyle_t fillStyle = config->fillStyle();
        switch(fillStyle){
            case ZEROS:
                GridFiller::fillEmptyGrid(grid);
                break;
            case RANDOM:
                GridFiller::fillGridRandomly(grid);
                break;
            case FROMFILE:
                GridFiller::fillGrid(grid,config->getGridFile());
                break;
        }
        
    } catch(exception& e){
        cerr << "Exception in grid filling: " << e.what() << endl;
        return -84;
    }
    
    return  0;
}

int main(int argc, char** argv) {
    
    if(argc < 2){
        cout << "ERROR: KEDF client must be called with a config file as the argument\n";
        return 1;
    }
    
    // read the config file
    Configuration* config = NULL;
    try{
        config = new Configuration(argv[1]);
    } catch(exception& e){
        cerr << "Exception in configuration parsing: " << e.what() << endl;
        return -21;
    }
    
    // construct the grid object and fill it
    const size_t xDim = config->getXDim();
    const size_t yDim = config->getYDim();
    const size_t zDim = config->getZDim();
    const shared_ptr<mat> cellVectors = config->getCellVectors();
    const string gridConfig = config->getGridConfig();
    const string kedfConfig = config->getKEDFConfig();
    
#ifdef _OPENMP
    fftw_init_threads();
#endif

    if(gridConfig.compare("fftw3,out-of-place") == 0){
        CartesianOOPGrid* smpGrid = new CartesianOOPGrid(xDim,yDim,zDim, cellVectors);
        KEDFFactory<CartesianOOPGrid>* factory = new KEDFFactory<CartesianOOPGrid>();
        KEDF<CartesianOOPGrid>* smpKEDF;
        try{
           smpKEDF = factory->constructKEDF(smpGrid, kedfConfig);
        } catch (exception e){
            cout << "KEDF is not a standard KEDF. Trying Fourier ones..." << endl;
            try{
                smpKEDF = factory->constructFourierKEDF(smpGrid, kedfConfig);
            } catch(exception e){
                cerr << "Also not a Fourier KEDF. Exiting." << endl;
                return -31418;
            }
        }

        smpGrid->getGNorms();
        
        const int ret = fillGrid(smpGrid, config);
        if(ret){
            return ret;
        }
        
        runKEDF(config,smpKEDF,smpGrid);
    
        delete smpKEDF;
        delete smpGrid;
        
        return 0;
#ifdef LIBKEDF_OCL
    } else if(gridConfig.compare("clfft,out-of-place") == 0){
        
        size_t platformNo = 0;
        size_t deviceNo = 0;
        
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
        
        CartesianOCLOOPGrid* oclGrid = (new CartesianOCLOOPGrid(xDim,yDim,zDim, cellVectors, platformNo, deviceNo));
        
        KEDFFactory<CartesianOCLOOPGrid>* factory = new KEDFFactory<CartesianOCLOOPGrid>();
        KEDF<CartesianOCLOOPGrid>* oclKEDF;
        try{
           oclKEDF = factory->constructKEDF(oclGrid, kedfConfig);
        } catch (exception e){
            cout << "KEDF is not a standard KEDF. Trying Fourier ones..." << endl;
            try{
                oclKEDF = factory->constructFourierKEDF(oclGrid, kedfConfig);
            } catch(exception e){
                cerr << "Also not a Fourier KEDF. Exiting." << endl;
                return -31418;
            }
        }

        oclGrid->getGNorms();
        
        const int ret = fillGrid(oclGrid, config);
        if(ret){
            return ret;
        }
        
        runKEDF(config,oclKEDF,oclGrid);
    
        delete oclKEDF;
        delete oclGrid;

        clfftTeardown();
        
        return 0;
#endif
    } else {
        throw runtime_error("Unknown Grid definition " + gridConfig);
    }
}
