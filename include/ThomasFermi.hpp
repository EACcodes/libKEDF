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

#ifndef THOMASFERMI_HPP
#define	THOMASFERMI_HPP

#include <cmath>
#include "KEDF.hpp"
#include "KEDFConstants.hpp"
#include "FourierGrid.hpp"
using namespace arma;
using namespace std;

template<class GridType>
class ThomasFermi: public KEDF<GridType> {
public:
    ThomasFermi(GridType* example){}
    
    ~ThomasFermi(){}
    
    string getMethodDescription() const {
        return "Thomas-Fermi KEDF";
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
    
    double calcEnergy(const GridType& grid) const {
        
        unique_ptr<GridType> workGrid(grid.duplicate());
        cube* density = workGrid->getRealGrid();
    
        // essentially pow(density,5/3)
        const uword elems = density->n_elem;
        #pragma omp parallel for default(none) shared(density)
        for(uword x = 0; x < elems; ++x){
            const double dens = density->at(x);
            if(dens < KEDF_CUTOFFDENSITY){
                density->at(x) = 0.0;
                continue;
            }
            const double densSq = dens*dens;
            const double dens23 = cbrt(densSq);
            const double dens53 = dens23*dens;
            density->at(x) = dens53;
        }
        workGrid->complete(density);

        const double energy = workGrid->integrate();
    
        return _CTF*energy;
    }
    
    double calcPotential(const GridType& grid, GridType& potential) const {
        
        unique_ptr<GridType> workGrid(grid.duplicate());
        cube* density = workGrid->getRealGrid();

        cube* potTens = potential.getRealGrid();

        const double preFactor = 5.0/3.0 * _CTF;

        const uword elems = density->n_elem;
        #pragma omp parallel for default(none) shared(density,potTens)
        for(uword x = 0; x < elems; ++x){
            const double dens = density->at(x);
            if(dens < KEDF_CUTOFFDENSITY){
                density->at(x) = 0.0;
                potTens->at(x) = 0.0;
                continue;
            }
            const double densSq = dens*dens;
            const double dens23 = cbrt(densSq);
            const double dens53 = dens23*dens;
            density->at(x) = dens53;
            potTens->at(x) = preFactor * dens23;
        }
        workGrid->complete(density);
        potential.complete(potTens);
    
        const double energy = workGrid->integrate();
    
        return _CTF*energy;
    }
    
    unique_ptr<StressTensor> calcStress(const GridType& grid) const {
        
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
};

#ifdef LIBKEDF_OCL
#include "ThomasFermiOCL.hpp"
#endif


#endif	/* THOMASFERMI_HPP */

