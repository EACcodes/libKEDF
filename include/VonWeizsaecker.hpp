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
#ifndef VONWEIZSAECKER_HPP
#define	VONWEIZSAECKER_HPP

#include <cmath>
#include "KEDFConstants.hpp"
#include "KEDF.hpp"
#include "Grid.hpp"
using namespace arma;
using namespace std;

template<class GridType>
class VonWeizsaecker: public KEDF<GridType> {
    
public:

    VonWeizsaecker(GridType* example){}
    
    ~VonWeizsaecker(){}
    
    string getMethodDescription() const {
        return "von Weizsaecker KEDF";
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
    
    double calcEnergy(const GridType& grid) const {
                
        unique_ptr<GridType> workGrid(grid.duplicate());
        workGrid->sqrtGrid();
        
        unique_ptr<GridType> laplacian(workGrid->laplacian());
        
        laplacian->multiplyElementwise(workGrid.get());

        const double energy = laplacian->integrate();

        return -0.5*energy;
    }
    
    double calcPotential(const GridType& grid, GridType& potential) const {
        
        unique_ptr<GridType> workGrid(grid.duplicate());
        workGrid->sqrtGrid();
    
        const cube* density = workGrid->readRealGrid();
    
        const unique_ptr<GridType> laplacian(workGrid->laplacian());
    
        cube* pot = potential.getRealGrid();
        const cube* lapcube = laplacian->readRealGrid();
        const double cutoffSq = sqrt(KEDF_CUTOFFDENSITY);
    
        const uword elems = density->n_elem;
        #pragma omp parallel for default(none) shared(pot,lapcube,density)
        for(uword x = 0; x < elems; ++x){
            const double dens = density->at(x);
            if(dens < cutoffSq){
                // cutoff to avoid divergence
                pot->at(x) = 0.0;
            } else {
                const double lap = lapcube->at(x);
                pot->at(x) = -0.5*lap/(dens);
            }
        }
        potential.complete(pot);
    
        laplacian->multiplyElementwise(workGrid.get());
        
        const double energy = laplacian->integrate();
    
        return -0.5*energy;
    }
    
    unique_ptr<StressTensor> calcStress(const GridType& grid) const {
        
        unique_ptr<StressTensor> stress = make_unique<StressTensor>();
        mat* const tensor = stress->getTensor();    

        unique_ptr<GridType> workGrid(grid.duplicate());
        workGrid->sqrtGrid();

        // we will need each directional divergence three times: store them it is worth the memory
        unique_ptr<GridType> directionalDivs[3];
        directionalDivs[0] = workGrid->directionalDivergenceX();
        directionalDivs[1] = workGrid->directionalDivergenceY();
        directionalDivs[2] = workGrid->directionalDivergenceZ();

        const unsigned long long totalPoints = workGrid->getTotalGridPoints();

        for(size_t i = 0; i < 3; ++i){

            const cube* dirI = directionalDivs[i]->readRealGrid();

            const uword elems = dirI->n_elem;

            for(size_t j = i; j < 3; ++j){

                const cube* dirJ = directionalDivs[j]->readRealGrid();

                // mix the directional derivatives together
                cube* realG = workGrid->getRealGrid();
                #pragma omp parallel for default(none) shared(realG, dirI, dirJ)
                for(uword x = 0; x < elems; ++x){

                    const double dI = dirI->at(x);
                    const double dJ = dirJ->at(x);

                    realG->at(x) = dI*dJ;
                }
                workGrid->complete(realG);

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
};

#ifdef LIBKEDF_OCL
#include "VonWeizsaeckerOCL.hpp"
#endif

#endif	/* VONWEIZSAECKER_HPP */
