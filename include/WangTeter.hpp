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

#ifndef WANGTETER_HPP
#define	WANGTETER_HPP

#include <armadillo>
#include <memory>
#include "KEDF.hpp"
#include "FourierGrid.hpp"
#include "HelperFunctions.hpp"
using namespace std;
using namespace arma;

template<class GridType>
class WangTeter: public KEDF<GridType> {
    
public:
    
    WangTeter(GridType* example, const double alpha, const double beta, cube* keKernel)
    : _alpha(alpha), _beta(beta), _keKernel(keKernel){
    }
    
    ~WangTeter() {
    }
    
    string getMethodDescription() const {
        return "Wang-Teter KEDF";
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
    
    double calcEnergy(const GridType& grid) const {

        unique_ptr<GridType> workGridB = grid.duplicate();
        
        workGridB->powGrid(_beta);
        
        cx_cube* recB = workGridB->getReciprocalGrid();
        
        const uword recElems = recB->n_elem;

        #pragma omp parallel for default(none) shared(recB)
        for (uword x = 0; x < recElems; ++x) {
            recB->at(x) *= _keKernel->at(x);
        }
        workGridB->completeReciprocal(recB);
        
        cube* realB = workGridB->getRealGrid();
        
        const uword elems = realB->n_elem;

        const cube* dens = grid.tryReadRealGrid();
        if (!dens) {
            // unfortunately, we need to get a copy of the grid
            unique_ptr<FourierGrid> workGrid = grid.createFourierDuplicate();
            dens = workGrid->readRealGrid();
        }

        #pragma omp parallel for default(none) shared(realB,dens)
        for (uword x = 0; x < elems; ++x) {
            realB->at(x) *= pow(dens->at(x), _alpha);
        }
        workGridB->complete(realB);        

        const double eWT = _CTF * workGridB->integrate();

        return eWT;
    }
    
    double calcPotential(const GridType& grid, GridType& potential) const {

        const cube* dens = grid.tryReadRealGrid();
        if (!dens) {
            // unfortunately, we need to get a copy of the grid
            unique_ptr<GridType> workGrid = grid.duplicate();
            dens = workGrid->readRealGrid();
        }

        unique_ptr<GridType> workGridA = grid.duplicate();
        cube* densityA = workGridA->getRealGrid();

        const uword elems = densityA->n_elem;
        const size_t nSlices = densityA->n_slices;
        const size_t nRows = densityA->n_rows;
        const size_t nCols = densityA->n_cols;

        auto densityB = MemoryFunctions::allocateScratch(nRows, nCols, nSlices);
        cube* poten = potential.getRealGrid();

        #pragma omp parallel for default(none) shared(densityA,densityB,poten)
        for (uword x = 0; x < elems; ++x) {
            const double rho = densityA->at(x);
            const double rhoPBM = pow(rho, (_beta - 1));
            densityB->at(x) = rhoPBM; // density B contains rho**(beta-1)
            poten->at(x) = rhoPBM*rho; // poten contains rho**(beta)
            densityA->at(x) = pow(rho, _alpha); // density A contains rho**(alpha)
        }
        potential.complete(poten);
        workGridA->complete(densityA);
        
        cx_cube* recPot = potential.getReciprocalGrid();
        
        const uword recElems = recPot->n_elem;

        #pragma omp parallel for default(none) shared(recPot)
        for (uword x = 0; x < recElems; ++x) {
            recPot->at(x) *= _keKernel->at(x);
        }
        potential.completeReciprocal(recPot);
        
        potential.multiplyElementwise(workGridA.get());

        // get the energy
        const double eWT = _CTF * potential.integrate();

        // transform what is currently in potential into the first part of the actual potential
        cube* realPot = potential.getRealGrid();

        const double preAl = _CTF*_alpha;
        #pragma omp parallel for default(none) shared(realPot,dens)
        for (uword x = 0; x < elems; ++x) {
            realPot->at(x) *= preAl / dens->at(x);
        }

        // now take what is in densityA and FFT it
        cx_cube* recDensA = workGridA->getReciprocalGrid();

        #pragma omp parallel for default(none) shared(recDensA)
        for (size_t x = 0; x < recElems; ++x) {
            recDensA->at(x) *= _keKernel->at(x);
        }
        workGridA->completeReciprocal(recDensA);

        densityA = workGridA->getRealGrid();

        const double preBe = _CTF*_beta;
        #pragma omp parallel for default(none) shared(realPot,densityA,densityB)
        for (size_t x = 0; x < elems; ++x) {
            const double pot = densityA->at(x) * preBe * densityB->at(x);
            realPot->at(x) += pot;
        }
        potential.complete(realPot);

        return eWT;
    }
    
    unique_ptr<StressTensor> calcStress(const GridType& grid) const {
        throw runtime_error("not yet implemented");
    }

private:

    const double _CTF = 2.87123400018819;
    const double _alpha;
    const double _beta;
    const cube * _keKernel;
};

class WangTeterKernel {
public:
    WangTeterKernel(const double alpha, const double beta, const double rho0, const double lambdaTF, const double muVW, const double ft = (5.0/3.0));
    ~WangTeterKernel();
    
    void fillWTKernelReciprocal(cube* kernel, const cube* gNorms);
    
private:
    double _alpha;
    double _beta;
    double _rho0;
    double _ft;
    double _lambda;
    double _mu;
    double _coeff;
    double _tkF;
};

#ifdef LIBKEDF_OCL
#include "WangTeterOCL.hpp"
#endif

#endif	/* WANGTETER_HPP */

