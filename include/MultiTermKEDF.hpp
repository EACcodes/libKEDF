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

/* 
 * Author: Johannes M Dieterich
 */

#ifndef MULTITERMKEDF_HPP
#define MULTITERMKEDF_HPP

#include <armadillo>
#include <memory>
#include <vector>
#include "KEDF.hpp"
using namespace std;
using namespace arma;

template<class GridType>
class MultiTermKEDF: public KEDF<GridType> {
public:
    MultiTermKEDF(GridType* example, unique_ptr<vector<shared_ptr<KEDF<GridType> > > > kedfs, unique_ptr<vector<double> > prefactors)
             : _kedfs(move(kedfs)), _prefacs(move(prefactors)){
    }

    ~MultiTermKEDF(){
    }
    
    string getMethodDescription() const override {
    
        const typename vector<shared_ptr<KEDF<GridType> > >::iterator kedfB = _kedfs->begin();
        const vector<double>::iterator facB = _prefacs->begin();
        const typename vector<shared_ptr<KEDF<GridType> > >::iterator kedfE = _kedfs->end();
        const vector<double>::iterator facE = _prefacs->end();
    
        typename vector<shared_ptr<KEDF<GridType> > >::iterator kedf;
        vector<double>::iterator fac;
        string descr = "";
        for(kedf = kedfB, fac = facB; kedf != kedfE && fac != facE; ++kedf, ++fac){
            const double myFac = *fac;
            const string s = to_string(myFac) + (*kedf)->getMethodDescription() + "\n";
            descr += s;
        }
    
        return descr;
    }
    
    vector<string> getCitations() const override {
    
        vector<string> totCitations(0);
    
        const typename vector<shared_ptr<KEDF<GridType> > >::iterator kedfB = _kedfs->begin();
        const typename vector<shared_ptr<KEDF<GridType> > >::iterator kedfE = _kedfs->end();
        for(typename vector<shared_ptr<KEDF<GridType> > >::iterator kedf = kedfB; kedf != kedfE; ++kedf){
        
            const vector<string> myCitations = (*kedf)->getCitations();
            totCitations.insert(end(totCitations), begin(myCitations), end(myCitations));
        }
    
        return totCitations;
    }
    
    vector<string> getWorkingEquations() const override {
    
        vector<string> totEquations(0);
    
        const typename vector<shared_ptr<KEDF<GridType> > >::iterator kedfB = _kedfs->begin();
        const typename vector<shared_ptr<KEDF<GridType> > >::iterator kedfE = _kedfs->end();
        for(typename vector<shared_ptr<KEDF<GridType> > >::iterator kedf = kedfB; kedf != kedfE; ++kedf){
        
            const vector<string> myEqs = (*kedf)->getWorkingEquations();
            totEquations.insert(end(totEquations), begin(myEqs), end(myEqs));
        }
    
        return totEquations;
    }
    
    double calcEnergy(const GridType& grid) const override {
    
        const typename vector<shared_ptr<KEDF<GridType> > >::iterator kedfB = _kedfs->begin();
        const vector<double>::iterator facB = _prefacs->begin();
        const typename vector<shared_ptr<KEDF<GridType> > >::iterator kedfE = _kedfs->end();
        const vector<double>::iterator facE = _prefacs->end();
    
        typename vector<shared_ptr<KEDF<GridType> > >::iterator kedf;
        vector<double>::iterator fac;
        double energy = 0.0;
        for(kedf = kedfB, fac = facB; kedf != kedfE && fac != facE; ++kedf, ++fac){
            const double myFac = *fac;
            const double myE = (*kedf)->calcEnergy(grid);
            energy += myFac*myE;
        }
    
        return energy;
    }

    
    double calcPotential(const GridType& grid, GridType& potential) const override  {
    
        const typename vector<shared_ptr<KEDF<GridType> > >::iterator kedfB = _kedfs->begin();
        const vector<double>::iterator facB = _prefacs->begin();
        const typename vector<shared_ptr<KEDF<GridType> > >::iterator kedfE = _kedfs->end();
        const vector<double>::iterator facE = _prefacs->end();
    
        unique_ptr<GridType> potCopy(potential.emptyDuplicate());
        cube *potCube = potential.getRealGrid();
        // zero the potential
        potCube->fill(0.0);
        potential.complete(potCube);
    
        typename vector<shared_ptr<KEDF<GridType> > >::iterator kedf;
        vector<double>::iterator fac;
        double energy = 0.0;
        for(kedf = kedfB, fac = facB; kedf != kedfE && fac != facE; ++kedf, ++fac){
            const double myFac = *fac;
            const double myE = (*kedf)->calcPotential(grid, *potCopy.get());
        
            potential.fmaGrid(myFac,potCopy.get());
            potential.finalize();
            energy += myFac*myE;
        }
    
        potCopy.reset();
        
        return energy;
    }
    
    unique_ptr<StressTensor> calcStress(const GridType& grid) const override {
    
        const typename vector<shared_ptr<KEDF<GridType> > >::iterator kedfB = _kedfs->begin();
        const vector<double>::iterator facB = _prefacs->begin();
        const typename vector<shared_ptr<KEDF<GridType> > >::iterator kedfE = _kedfs->end();
        const vector<double>::iterator facE = _prefacs->end();
    
        unique_ptr<StressTensor> stress = make_unique<StressTensor>();
        mat* const tensor = stress->getTensor();
    
        typename vector<shared_ptr<KEDF<GridType> > >::iterator kedf;
        vector<double>::iterator fac;
        for(kedf = kedfB, fac = facB; kedf != kedfE && fac != facE; ++kedf, ++fac){
            const double myFac = *fac;
            const unique_ptr<StressTensor> myStress = (*kedf)->calcStress(grid);
            const mat * const myTensor = myStress->getTensor();
        
            tensor->at(0,0) = myFac*myTensor->at(0,0);
            tensor->at(1,0) = myFac*myTensor->at(1,0);
            tensor->at(2,0) = myFac*myTensor->at(2,0);
            tensor->at(0,1) = myFac*myTensor->at(0,1);
            tensor->at(1,1) = myFac*myTensor->at(1,1);
            tensor->at(2,1) = myFac*myTensor->at(2,1);
            tensor->at(0,2) = myFac*myTensor->at(0,2);
            tensor->at(1,2) = myFac*myTensor->at(1,2);
            tensor->at(2,2) = myFac*myTensor->at(2,2);
        }
    
        return stress;
    }
    
private:
    unique_ptr<vector<shared_ptr<KEDF<GridType> > > > _kedfs;
    unique_ptr<vector<double> > _prefacs;
};

#endif /* MULTITERMKEDF_HPP */

