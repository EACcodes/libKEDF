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

#ifndef HUANGCARTER_HPP
#define HUANGCARTER_HPP

#include <cmath>
#include <stdio.h>
#include "FourierGrid.hpp"
#include "HelperFunctions.hpp"
#include "WangGovindCarter.hpp"
using namespace arma;
using namespace std;

class HuangCarterODE : public ODEKernel {
    
public:
    HuangCarterODE(const double beta);
    ~HuangCarterODE();
    void evaluate(const double t, double y[], double yp[]) override;
    
private:
    const double _beta;
};

template<class GridType>
class HuangCarter: public KEDF<GridType> {

public:
    
    HuangCarter(GridType* example, const size_t numEta, const double etaStep, unique_ptr<vec> w, unique_ptr<vec> w1,
            const double alpha, const double beta, const double lambda,
            const double rhoS, const double c, const double refRatio,
            const double cutoffDens, const bool trashEmptyBins, const size_t cutoffEmptyBin, const bool doInterPolation = false)
    : _numEta(numEta), _etaStep(etaStep), _doInterPolation(doInterPolation), _w(move(w)), _w1(move(w1)), _alpha(alpha), _beta(beta), _hc_lambda(lambda),
            _rhoS(rhoS), _c(c), _midXi(cbrt(3.0*M_PI*M_PI*_rhoS)), _refRatio(refRatio),
            _cutoffDens(cutoffDens), _trashEmptyBins(trashEmptyBins), _cutoffEmptyBin(cutoffEmptyBin) {
        
        // split the w/w' tables up into two: one more used, smaller one and one less used, larger one
        const size_t highSize = _numEta - _KERNELSPLIT + 1; // +1 to have some overlap in the case that ind == KERNELSPLIT-1, ind+1 == KERNELSPLIT
        _wLow = make_unique<vec>(_KERNELSPLIT);
        _wHigh = make_unique<vec>(highSize);
        _w1Low = make_unique<vec>(_KERNELSPLIT);
        _w1High = make_unique<vec>(highSize);
        
        const size_t lowLength = _KERNELSPLIT*sizeof(double);
        const size_t highLength = highSize*sizeof(double);
        
        double* wP = _w->memptr();
        double* w1P = _w1->memptr();
        
        double* wLP = _wLow->memptr();
        double* wHP = _wHigh->memptr();
        double* w1LP = _w1Low->memptr();
        double* w1HP = _w1High->memptr();
        
        memcpy(wLP, wP, lowLength);
        memcpy(wHP, &wP[_KERNELSPLIT-1], highLength);
        memcpy(w1LP, w1P, lowLength);
        memcpy(w1HP, &w1P[_KERNELSPLIT-1], highLength);
        
    }
    
    ~HuangCarter(){
    }
    
    string getMethodDescription() const {
        return "Huang-Carter (2010) KEDF";
    }
    
    vector<string> getCitations() const {
        
        vector<string> citations(0);
        citations.push_back("C. Huang and E. A. Carter, Phys. Rev. B 81, 045206 (2010).");

        return citations;
    }

    vector<string> getWorkingEquations() const {
        vector<string> citations(0);
        citations.push_back("sorry, not yet");
    
        return citations;
    }
    

    double calcEnergy(const GridType& grid) const {
        
        const cube* dens = grid.tryReadRealGrid();
        if (!dens) {
            // unfortunately, we need to get a copy of the grid
            unique_ptr<GridType> workGrid = grid.duplicate();
            dens = workGrid->readRealGrid();
        }
        
        unique_ptr<GridType> workGridA = grid.duplicate();
        
        cube* densA = workGridA->getRealGrid();
   
        const uword elems = densA->n_elem;
        const size_t nSlices = densA->n_slices;
        const size_t nRows = densA->n_rows;
        const size_t nCols = densA->n_cols;
        
        workGridA->complete(densA);
        
        unique_ptr<GridType> workGridXi = grid.emptyDuplicate();
        cube* xiT = workGridXi->getRealGrid();
        calcXi(workGridA.get(), xiT);
        workGridXi->complete(xiT);
        
        double minXi = 0.0;
        double maxXi = 0.0;
        workGridXi->minMax(minXi, maxXi);
        
        const cube* xi = workGridXi->readRealGrid();
        
        // this allocation happens after calcXi since that one internally needs to allocate a grid to hold the gradient
        unique_ptr<GridType> workGridB = grid.emptyDuplicate();
        unique_ptr<GridType> workGridC = grid.emptyDuplicate();
        
        const double upper = ceil(log(maxXi/_midXi) / log(_refRatio));
        const double lower = floor(log(minXi/_midXi) / log(_refRatio));
        const size_t nBins = round(upper-lower)+1;        
        
        // get the binning array
        auto bins = MemoryFunctions::allocateScratch(nBins);
        
        for(size_t i = 0; i < nBins; ++i){
            bins->at(i) = _midXi * pow(_refRatio,(lower + i));
        }
        
        workGridA->powGrid(_alpha);
        densA = workGridA->getRealGrid();
        
        unique_ptr<GridType> binnedDens = grid.emptyDuplicate();
        cube* binned = binnedDens->getRealGrid();
        
        unique_ptr<GridType> result = grid.emptyDuplicate();
        cx_cube* resRec = result->getReciprocalGrid();
        
        const uword recElems = resRec->n_elem;
        const size_t nSlicesRec = resRec->n_slices;
        const size_t nRowsRec = resRec->n_rows;
        const size_t nColsRec = resRec->n_cols;
        
        // scratch space for the kernels
        auto k0 = MemoryFunctions::allocateScratch(nRowsRec, nColsRec, nSlicesRec);
        auto k1 = MemoryFunctions::allocateScratch(nRowsRec, nColsRec, nSlicesRec);
        
        const cube* gNorms = workGridA->getGNorms();
        
        // go over all the bins
        for(size_t bin = 0; bin < nBins-1; ++bin){
            
            const double binI = bins->at(bin);
            const double binI1 = bins->at(bin+1);        
            const double incr = binI1 - binI;
            
            // get the grid points with density value in this bin, count how many there are, hence: can't be fully parallel!
#ifdef _OPENMP
            size_t nonZeroArr[nSlices];
            #pragma omp parallel for default(none) shared(nonZeroArr,xi,dens,densA,binned)
#else
            // this needs to be a size_t[] in the parallel case
            size_t nonZero = 0;
#endif
            for(size_t x = 0; x < nSlices; ++x){
#ifdef _OPENMP
                nonZeroArr[x] = 0;
#endif
                for(size_t col = 0; col < nCols; ++col){
                    for(size_t row = 0; row < nRows; ++row){
                        
                        const double myXI = xi->at(row,col,x);
                        if(myXI < binI || myXI >= binI1 || dens->at(row,col,x) < _cutoffDens){
                            binned->at(row,col,x) = 0.0;
                        } else {
                            binned->at(row,col,x) = densA->at(row,col,x);
#ifdef _OPENMP
                            ++(nonZeroArr[x]);
#else
                            ++nonZero;
#endif
                        }
                    }
                }
            }
            
#ifdef _OPENMP
            // do the aggregation
            size_t nonZero = 0;
            for(size_t x = 0; x < nSlices; ++x){
                nonZero += nonZeroArr[x];
            }
#endif

            if(_trashEmptyBins && nonZero <= _cutoffEmptyBin){
#ifdef LIBKEDF_DEBUG
                cout << "DEBUG: Skipping bin " << bin << " because " << nonZero << endl;
#endif
                continue;
            }
            
            // first round of kernels 
            if(bin == 0){
                // we only need to compute these kernels in the first iteration
                // subsequent iterations can simple use the older ones
                calcHC10Kernels(binI, k0.get(), k1.get(), gNorms);
            }
                                    
            // assemble first part of the "energy grid"
            workGridB->resetToReal();
            workGridC->resetToReal();
            cube* temp = workGridB->getRealGrid();
            cube* temp2 = workGridC->getRealGrid();
            
            #pragma omp parallel for default(none) shared(xi,temp,temp2,binned)
            for(uword x = 0; x < elems; ++x){
                        
                // assemble h0 and h1 first
                const double t = (xi->at(x) - binI) / incr;
                const double tSq = t*t;
                const double t3 = tSq*t;
                
                const double h0 = 2*t3 - 3*tSq + 1;
                const double h1 = t3 - 2*tSq + t;
                
                // now create the intermediate data
                temp->at(x) = binned->at(x) * h0;
                temp2->at(x) = binned->at(x) * h1;
            }
            workGridB->complete(temp);
            workGridC->complete(temp2);
            
            // get it to g space
            const cx_cube* tempRec = workGridB->readReciprocalGrid();
            const cx_cube* tempRec2 = workGridC->readReciprocalGrid();
            
            if(bin == 0){
                #pragma omp parallel for default(none) shared(resRec,tempRec,tempRec2,k0,k1)
                for(uword x = 0; x < recElems; ++x){
                    resRec->at(x) = k0->at(x) * tempRec->at(x)
                            + incr* (k1->at(x)/binI) * tempRec2->at(x);
                }
            } else {
                #pragma omp parallel for default(none) shared(resRec,tempRec,tempRec2,k0,k1)
                for(uword x = 0; x < recElems; ++x){
                    resRec->at(x) += k0->at(x) * tempRec->at(x)
                            + incr* (k1->at(x)/binI) * tempRec2->at(x);
                }
            }
            
            // second round of kernels
            calcHC10Kernels(binI1, k0.get(), k1.get(), gNorms);
                        
            // assemble second part of the "energy grid"
            workGridB->resetToReal();
            workGridC->resetToReal();
            
            temp = workGridB->getRealGrid();
            temp2 = workGridC->getRealGrid();
            
            #pragma omp parallel for default(none) shared(xi,temp,temp2,binned)
            for(uword x = 0; x < elems; ++x){
                        
                // assemble h0 and h1 again
                const double t = (xi->at(x) - binI) / incr;
                const double tSq = t*t;
                const double t3 = tSq*t;
                
                const double h0 = -2*t3 + 3*tSq;
                const double h1 = t3 - tSq;
                
                temp->at(x) = binned->at(x) * h0;
                temp2->at(x) = binned->at(x) * h1;
            }
            workGridB->complete(temp);
            workGridC->complete(temp2);
            
            // get it to g space
            const cx_cube* tempRec3 = workGridB->readReciprocalGrid();
            const cx_cube* tempRec4 = workGridC->readReciprocalGrid();
            
            #pragma omp parallel for default(none) shared(resRec,tempRec3,tempRec4,k0,k1)
            for(uword x = 0; x < recElems; ++x){
                resRec->at(x) += k0->at(x) * tempRec3->at(x)
                        + incr * (k1->at(x)/binI1) * tempRec4->at(x);
            }            
        }
        workGridA->complete(densA);        
        workGridA.reset();
        workGridXi.reset();
        result->completeReciprocal(resRec);
        
        cube* res = result->getRealGrid();
      
        #pragma omp parallel for default(none) shared(res, dens)
        for(uword x = 0; x < elems; ++x){
            res->at(x) *= pow(dens->at(x),_beta);
        }
        result->complete(res);
        
        const double integral = result->integrate();
        
        return _CTF*_c*integral;
    }
    
    double calcPotential(const GridType& grid, GridType& potential) const {
        
        const cube* dens = grid.tryReadRealGrid();
        if (!dens) {
            // unfortunately, we need to get a copy of the grid
            unique_ptr<GridType> workGrid = grid.duplicate();
            dens = workGrid->readRealGrid();
        }
        
        unique_ptr<GridType> workGridA = grid.duplicate();
        
        cube* densA = workGridA->getRealGrid();
        
        const uword elems = densA->n_elem;
        const size_t nSlices = densA->n_slices;
        const size_t nRows = densA->n_rows;
        const size_t nCols = densA->n_cols;
        
        workGridA->complete(densA);
        
        unique_ptr<GridType> workGridXi = grid.emptyDuplicate();
        cube* xiT = workGridXi->getRealGrid();
        
        auto ss = MemoryFunctions::allocateScratch(nRows, nCols, nSlices);
                
        calcXiAndSS(workGridA.get(), xiT, ss.get());
        workGridXi->complete(xiT);
        
        double minXi = 0.0;
        double maxXi = 0.0;
        workGridXi->minMax(minXi, maxXi);
        
        const cube* xi = workGridXi->readRealGrid();
        
        // this allocation happens after calcXi since that one internally needs to allocate a grid to hold the gradient
        unique_ptr<GridType> workGridB = grid.emptyDuplicate();
        unique_ptr<GridType> workGridC = grid.emptyDuplicate();
        
        cube* pot = potential.getRealGrid();
        unique_ptr<GridType> workGridBeta = grid.emptyDuplicate();
        cube* densBReal = workGridBeta->getRealGrid();
        
        #pragma omp parallel for default(none) shared(pot,dens,densBReal)
        for(uword x = 0; x < elems; ++x){
            
            const double d = dens->at(x);
            const double densBM1 = pow(d,_beta-1);
            const double densB = densBM1 * d;
            
            pot->at(x) = densBM1;
            densBReal->at(x) = densB;
        }
        workGridBeta->complete(densBReal);
        
        const cx_cube* densB = workGridBeta->readReciprocalGrid();
        
        unique_ptr<GridType> fkf0 = grid.emptyDuplicate();
        unique_ptr<GridType> fk1rf = grid.emptyDuplicate();
        
        const double upper = ceil(log(maxXi/_midXi) / log(_refRatio));
        const double lower = floor(log(minXi/_midXi) / log(_refRatio));
        const size_t nBins = round(upper-lower)+1;        
        
        // get the binning array
        auto bins = MemoryFunctions::allocateScratch(nBins);
        
        for(size_t i = 0; i < nBins; ++i){
            bins->at(i) = _midXi * pow(_refRatio,(lower + i));
        }
        
        workGridA->powGrid(_alpha);
        densA = workGridA->getRealGrid();
        
        unique_ptr<GridType> binnedDens = grid.emptyDuplicate();
        cube* binned = binnedDens->getRealGrid();
        
        unique_ptr<GridType> result = grid.emptyDuplicate();
        cx_cube* resRec = result->getReciprocalGrid();
        
        const uword recElems = resRec->n_elem;
        const size_t nSlicesRec = resRec->n_slices;
        const size_t nRowsRec = resRec->n_rows;
        const size_t nColsRec = resRec->n_cols;
        
        // scratch space for the kernels
        auto k0 = MemoryFunctions::allocateScratch(nRowsRec, nColsRec, nSlicesRec);
        auto k1 = MemoryFunctions::allocateScratch(nRowsRec, nColsRec, nSlicesRec);
        
        // other scratch space
        auto intpot = MemoryFunctions::allocateScratch(nRows, nCols, nSlices);
        auto pottmp = MemoryFunctions::allocateScratch(nRows, nCols, nSlices);
        
        const cube* gNorms = workGridA->getGNorms();
        
        // go over all the bins
        for(size_t bin = 0; bin < nBins-1; ++bin){
            
            const double binI = bins->at(bin);
            const double binI1 = bins->at(bin+1);        
            const double incr = binI1 - binI;
            
            // get the grid points with density value in this bin, count how many there are, hence: can't be fully parallel!
#ifdef _OPENMP
            size_t nonZeroArr[nSlices];
            #pragma omp parallel for default(none) shared(nonZeroArr,xi,dens,densA,binned)
#else
            // this needs to be a size_t[] in the parallel case
            size_t nonZero = 0;
#endif
            for(size_t x = 0; x < nSlices; ++x){
#ifdef _OPENMP
                nonZeroArr[x] = 0;
#endif
                for(size_t col = 0; col < nCols; ++col){
                    for(size_t row = 0; row < nRows; ++row){
                        
                        const double myXI = xi->at(row,col,x);
                        if(myXI < binI || myXI >= binI1 || dens->at(row,col,x) < _cutoffDens){
                            binned->at(row,col,x) = 0.0;
                        } else {
                            binned->at(row,col,x) = densA->at(row,col,x);
#ifdef _OPENMP
                            ++(nonZeroArr[x]);
#else
                            ++nonZero;
#endif
                        }
                    }
                }
            }
            
#ifdef _OPENMP
            // do the aggregation
            size_t nonZero = 0;
            for(size_t x = 0; x < nSlices; ++x){
                nonZero += nonZeroArr[x];
            }
#endif

            if(_trashEmptyBins && nonZero <= _cutoffEmptyBin){
#ifdef LIBKEDF_DEBUG
                cout << "DEBUG: Skipping bin " << bin << " because " << nonZero << endl;
#endif
                continue;
            }
            
            // first round of kernels 
            if(bin == 0){
                // we only need to compute these kernels in the first iteration
                // subsequent iterations can simple use the older ones
                calcHC10Kernels(binI, k0.get(), k1.get(), gNorms);
            
                fkf0->resetToReciprocal();
                cx_cube* fkf0Reci = fkf0->getReciprocalGrid();
                fk1rf->resetToReciprocal();
                cx_cube* fk1rfReci = fk1rf->getReciprocalGrid();
            
                #pragma omp parallel for default(none) shared(fk1rfReci,fkf0Reci,densB,k0,k1)
                for(uword x = 0; x < recElems; ++x){
                    fkf0Reci->at(x) = k0->at(x) * densB->at(x);
                    const double kref1 = k1->at(x) / binI;
                    fk1rfReci->at(x) = kref1 * densB->at(x);
                }
                fkf0->completeReciprocal(fkf0Reci);
                fk1rf->completeReciprocal(fk1rfReci);
            }
            
            const cube* fkf0Real = fkf0->readRealGrid();
            const cube* fk1rfReal = fk1rf->readRealGrid();

            // assemble first part of the "energy grid"
            workGridB->resetToReal();
            workGridC->resetToReal();
            cube* temp = workGridB->getRealGrid();
            cube* temp2 = workGridC->getRealGrid();
            
            #pragma omp parallel for default(none) shared(xi,temp,temp2,binned,dens,intpot,pottmp,fkf0Real,fk1rfReal, bin)
            for(uword x = 0; x < elems; ++x){
                
                // assemble h0 and h1 first
                const double t = (xi->at(x) - binI) / incr;
                const double tSq = t*t;
                const double t3 = tSq*t;
                
                const double h0 = 2*t3 - 3*tSq + 1;
                const double h1 = t3 - 2*tSq + t;
                
                // now create the intermediate data
                temp->at(x) = binned->at(x) * h0;
                temp2->at(x) = binned->at(x) * h1;
                
                const double myXI = xi->at(x);
                if(myXI < binI || myXI >= binI1 || dens->at(x) < _cutoffDens){
                    continue;
                }
                
                const double h00 = 6*tSq - 6*t;
                const double h10 = 3*tSq - 4*t + 1.0;
                
                // update the intpot
                const double myIntpot = h0 * fkf0Real->at(x) + incr * (h1 * fk1rfReal->at(x));
                intpot->at(x) = (bin == 0) ? myIntpot : intpot->at(x) + myIntpot;
                
                // update the pottmp
                const double myPottmp = (h00 * fkf0Real->at(x)) / incr + h10 * fk1rfReal->at(x);
                pottmp->at(x) = (bin == 0) ? myPottmp : pottmp->at(x) + myPottmp;
            }
            workGridB->complete(temp);
            workGridC->complete(temp2);
            
            // get it to g space
            const cx_cube* tempRec = workGridB->readReciprocalGrid();
            const cx_cube* tempRec2 = workGridC->readReciprocalGrid();

            if(bin == 0){
                #pragma omp parallel for default(none) shared(resRec,tempRec,tempRec2,k0,k1)
                for(uword x = 0; x < recElems; ++x){
                    resRec->at(x) = k0->at(x) * tempRec->at(x)
                            + incr* (k1->at(x)/binI) * tempRec2->at(x);
                }
            } else {
                #pragma omp parallel for default(none) shared(resRec,tempRec,tempRec2,k0,k1)
                for(uword x = 0; x < recElems; ++x){
                    resRec->at(x) += k0->at(x) * tempRec->at(x)
                            + incr* (k1->at(x)/binI) * tempRec2->at(x);
                }
            }
                        
            // second round of kernels
            calcHC10Kernels(binI1, k0.get(), k1.get(), gNorms);
            
            fkf0->resetToReciprocal();
            cx_cube* fkf0Reci2 = fkf0->getReciprocalGrid();
            fk1rf->resetToReciprocal();
            cx_cube* fk1rfReci2 = fk1rf->getReciprocalGrid();
            
            #pragma omp parallel for default(none) shared(fk1rfReci2,fkf0Reci2,densB,k0,k1)
            for(uword x = 0; x < recElems; ++x){
                fkf0Reci2->at(x) = k0->at(x) * densB->at(x);
                const double kref1 = k1->at(x) / binI1;
                fk1rfReci2->at(x) = kref1 * densB->at(x);
            }
            fkf0->completeReciprocal(fkf0Reci2);
            fk1rf->completeReciprocal(fk1rfReci2);
            
            const cube* fkf0Real2 = fkf0->readRealGrid();
            const cube* fk1rfReal2 = fk1rf->readRealGrid();
                        
            // assemble second part of the "energy grid"
            workGridB->resetToReal();
            workGridC->resetToReal();
            
            temp = workGridB->getRealGrid();
            temp2 = workGridC->getRealGrid();
            
            #pragma omp parallel for default(none) shared(xi,temp,temp2,binned,intpot,pottmp,fkf0Real2,fk1rfReal2,dens)
            for(uword x = 0; x < elems; ++x){
                
                // assemble h0 and h1 again
                const double t = (xi->at(x) - binI) / incr;
                const double tSq = t*t;
                const double t3 = tSq*t;
                
                const double h0 = -2*t3 + 3*tSq;
                const double h1 = t3 - tSq;
                
                temp->at(x) = binned->at(x) * h0;
                temp2->at(x) = binned->at(x) * h1;
                
                const double myXI = xi->at(x);
                if(myXI < binI || myXI >= binI1 || dens->at(x) < _cutoffDens){
                    continue;
                }
                
                const double h10 = 3*tSq - 4*t + 1.0;
                const double h01 = -6*tSq + 6*t;
                const double h11 = h10 + 2*t - 1.0;
                
                // update the intpot again
                intpot->at(x) += h0 * fkf0Real2->at(x) + incr * (h1 * fk1rfReal2->at(x));
                
                // update the pottmp
                pottmp->at(x) += (h01 * fkf0Real2->at(x))/incr + h11 * fk1rfReal2->at(x);
            }
            workGridB->complete(temp);
            workGridC->complete(temp2);
            
            // get it to g space
            const cx_cube* tempRec3 = workGridB->readReciprocalGrid();
            const cx_cube* tempRec4 = workGridC->readReciprocalGrid();
            
            #pragma omp parallel for default(none) shared(resRec,tempRec3,tempRec4,k0,k1)
            for(uword x = 0; x < recElems; ++x){
                resRec->at(x) += k0->at(x) * tempRec3->at(x)
                        + incr * (k1->at(x)/binI1) * tempRec4->at(x);
            }            
        }
        result->completeReciprocal(resRec);
        
        const double preFactor = _CTF*_c;
        
        cube* res = result->getRealGrid();
      
        #pragma omp parallel for default(none) shared(res, dens, densA, pot, ss, pottmp, intpot)
        for(uword x = 0; x < elems; ++x){
            const double d = dens->at(x);
            const double densBM1 = pot->at(x);
            const double dA = densA->at(x);
            
            const double tmp = res->at(x);
            const double tmp2 = densBM1 * tmp;
            
            const double potTmp2 = pottmp->at(x) * dA;
            
            const double kF = cbrt(3*M_PI*M_PI*d);
            const double pxiprho = kF/(3*d)*(1-7*_hc_lambda*ss->at(x));
            
            const double potPart1 = _alpha*dA/d*intpot->at(x);
            const double potPart2 = _beta*tmp2;
            const double potPart3 = pxiprho * potTmp2;
            
            const double rhoSq = d*d;
            const double rho4 = rhoSq*rhoSq;
            const double rho8 = rho4*rho4;
            const double rho83 = cbrt(rho8);
            
            const double t3 = (kF * _hc_lambda * 2 / rho83) * potTmp2;
            ss->at(x) = t3; // we are reusing ss here to safe some memory
            
            // only now multiply again with the density
            res->at(x) = tmp2 * d;
            
            pot->at(x) = potPart1 + potPart2 + potPart3; // at this point we are missing the big 4th contribution
        }
        result->complete(res);
        workGridA->complete(densA);
        
        intpot.reset();
        workGridXi.reset();
        
        // get the grid into reciprocal space
        workGridA->copyStateIn(&grid);
        cx_cube* recDens = workGridA->getReciprocalGrid();
        workGridA->completeReciprocal(recDens);
        
        workGridB->copyStateIn(workGridA.get());
        workGridB->multiplyGVectorsX();
        cube* gridX = workGridB->getRealGrid();
        
        #pragma omp parallel for default(none) shared(gridX, ss)
        for(uword x = 0; x < elems; ++x){
            gridX->at(x) *= ss->at(x);
        }        
        workGridB->complete(gridX);
        
        workGridB->multiplyGVectorsX();
        gridX = workGridB->getRealGrid();
        
        #pragma omp parallel for default(none) shared(pot, gridX)
        for(uword x = 0; x < elems; ++x){
            pot->at(x) -= gridX->at(x);
        }
        workGridB->complete(gridX);
        
        workGridB->copyStateIn(workGridA.get());
        workGridB->multiplyGVectorsY();
        cube* gridY = workGridB->getRealGrid();
        
        #pragma omp parallel for default(none) shared(gridY, ss)
        for(uword x = 0; x < elems; ++x){
            gridY->at(x) *= ss->at(x);
        }
        workGridB->complete(gridY);
        
        workGridB->multiplyGVectorsY();
        gridY = workGridB->getRealGrid();
        
        #pragma omp parallel for default(none) shared(pot, gridY)
        for(uword x = 0; x < elems; ++x){
            pot->at(x) -= gridY->at(x);
        }
        workGridB->complete(gridY);
        
        workGridB->copyStateIn(workGridA.get());
        workGridB->multiplyGVectorsZ();
        cube* gridZ = workGridB->getRealGrid();
        
        #pragma omp parallel for default(none) shared(gridZ, ss)
        for(uword x = 0; x < elems; ++x){
            gridZ->at(x) *= ss->at(x);
        }        
        workGridB->complete(gridZ);
        
        workGridB->multiplyGVectorsZ();
        gridZ = workGridB->getRealGrid();
        
        #pragma omp parallel for default(none) shared(pot, gridZ)
        for(uword x = 0; x < elems; ++x){
            pot->at(x) -= gridZ->at(x);
            pot->at(x) *= preFactor;
        }
        workGridB->complete(gridZ);
        workGridB.reset();
        
        potential.complete(pot);
        
        const double integral = result->integrate();
        
        return preFactor*integral;
    }
    
    unique_ptr<StressTensor> calcStress(const GridType& grid) const {
        throw runtime_error("XXX implement me");
    }
    
private:
    
    const size_t _numEta;
    const double _etaStep;
    const bool _doInterPolation;
    
    unique_ptr<vec> _w;
    unique_ptr<vec> _wLow;
    unique_ptr<vec> _wHigh;
    
    unique_ptr<vec> _w1;
    unique_ptr<vec> _w1Low;
    unique_ptr<vec> _w1High;
    
    // this setting relies on the fact that w/w' as handed over are defined in [0.0,50.0] with an increment of 0.001
    // empirically, we observed that the requested values are typically in [0,10]
    const size_t _KERNELSPLIT = 10000;
    
    const double _CUTXIHIGH = 1000.0;
    const double _CUTXILOW = 0.001;
    const double _CTF = 2.87123400018819;
    
    const double _alpha;
    const double _beta;
    const double _hc_lambda;
    const double _rhoS;
    const double _c;
    const double _midXi;
    const double _refRatio;
    const double _cutoffDens;
    const bool _trashEmptyBins;
    const size_t _cutoffEmptyBin;
    
    void calcXi(GridType* density, cube* xiReal) const {
        
        unique_ptr<GridType> gradientSquared(density->gradientSquared());
        
        const cube* realG = gradientSquared->readRealGrid();
        const cube* densReal = density->readRealGrid();
        
        const uword elems = xiReal->n_elem;
        
        #pragma omp parallel for default(none) shared(realG, densReal, xiReal)
        for(uword x = 0; x < elems; ++x){
            const double rho = densReal->at(x);
            const double rhoSq = rho*rho;
            const double rho4 = rhoSq*rhoSq;
            const double rho8 = rho4*rho4;
            const double rho83 = cbrt(rho8);
            
            const double ss = realG->at(x) / rho83;
            
            const double t = 3*M_PI*M_PI*rho;
            const double t13 = cbrt(t);
            
            const double xiE = t13 * (1 + _hc_lambda * ss);
            
            if(xiE > _CUTXIHIGH){
                xiReal->at(x) = _CUTXIHIGH;
            } else if(xiE < _CUTXILOW){
                xiReal->at(x) = _CUTXILOW;
            } else {
                xiReal->at(x) = xiE;
            }
        }

        gradientSquared.reset();
    }
    
    void calcXiAndSS(GridType* density, cube* xiReal, cube* ssReal) const {
        
        // so, in theory we could keep all gradient contributions in memory
        // and re-use here. BUT: I want to first do it this way and (yes)
        // redundantly compute this but since this is not done in the bin loop.
        // it's not as important and it saves 3 grids worth of memory
        // same holds true for pow(rho,8/3) but again, I think this can be
        // done smarter
        unique_ptr<GridType> gradientSquared(density->gradientSquared());
        
        const cube* realG = gradientSquared->readRealGrid();
        const cube* densReal = density->readRealGrid();
        
        const uword elems = xiReal->n_elem;
        
        #pragma omp parallel for default(none) shared(realG, densReal, xiReal, ssReal)
        for(uword x = 0; x < elems; ++x){
            const double rho = densReal->at(x);
            const double rhoSq = rho*rho;
            const double rho4 = rhoSq*rhoSq;
            const double rho8 = rho4*rho4;
            const double rho83 = cbrt(rho8);
            
            const double ss = realG->at(x) / rho83;
            ssReal->at(x) = ss;
            
            const double t = 3*M_PI*M_PI*rho;
            const double t13 = cbrt(t);
            
            const double xiE = t13 * (1 + _hc_lambda * ss);
            
            if(xiE > _CUTXIHIGH){
                xiReal->at(x) = _CUTXIHIGH;
            } else if(xiE < _CUTXILOW){
                xiReal->at(x) = _CUTXILOW;
            } else {
                xiReal->at(x) = xiE;
            }
        }

        gradientSquared.reset();
    }
    
    // XXX important: I think we can solve the performance impact of this almost completely
    // 1) going to two lookups (ind is typically below 10,000)
    // 2) most of the features are >10 -> keep this dense, make the other one sparser
    // 3) doing *actually* linear interpolation between points
    // 4) maybe 3) will allow to go more sparse for *both* tables
    // so, w/o 4) we would have 1-10 at 10,000 points -> 80k
    //            and 10-50 at, say, 20,000 points -> 160k
    // if 4) allows us a factor 2-5 reduction -> 40-16k for the smaller one
    //            and 80-32k for the bigger one.
    void calcHC10Kernels(const double xi, cube* k0, cube* k1, const cube* gNorms) const {
        
        const uword recElems = k0->n_elem;
        
        const double xi3 = xi*xi*xi;
        const double t8xi3 = 8*xi3;
        
        #pragma omp parallel for default(none) shared(k0, k1, gNorms)
        for(uword x = 0; x < recElems; ++x){
            
            const double eta = gNorms->at(x) / (2 * xi);
            const double fullInd = eta/_etaStep;
            const size_t ind = floor(fullInd);
            
            // check if our index is within our lookup table
            double tmp;
            double tmp1;
            if(ind >= _numEta-1){
                tmp = _w->at(_numEta-1);
                tmp1 = _w1->at(_numEta-1);
#ifndef LIBKEDF_ILIKESLOWCODE
            } else if(ind < _KERNELSPLIT-1){
                if(_doInterPolation){
                    const double rest = fullInd - ind; // will give how far ABOVE ind this actually was
                
                    const double wL = _wLow->at(ind);
                    const double wH = _wLow->at(ind + 1);
                    tmp = wL + rest*(wH-wL);
                
                    const double w1L = _w1Low->at(ind);
                    const double w1H = _w1Low->at(ind + 1);
                    tmp1 = w1L + rest*(w1H-w1L);                        
                } else {
                    tmp = (_wLow->at(ind) + _wLow->at(ind + 1)) * 0.5;
                    tmp1 = (_w1Low->at(ind) + _w1Low->at(ind + 1) ) * 0.5;
                }
            } else {
                const size_t ind2 = ind - _KERNELSPLIT + 1; // we are shifting one up
                if(_doInterPolation){
                    const double rest = fullInd - ind; // will give how far ABOVE ind this actually was
                
                    const double wL = _wHigh->at(ind2);
                    const double wH = _wHigh->at(ind2 + 1);
                    tmp = wL + rest*(wH-wL);
                
                    const double w1L = _w1High->at(ind2);
                    const double w1H = _w1High->at(ind2 + 1);
                    tmp1 = w1L + rest*(w1H-w1L);
                } else {
                    tmp = (_wHigh->at(ind2) + _wHigh->at(ind2 + 1)) * 0.5;
                    tmp1 = (_w1High->at(ind2) + _w1High->at(ind2 + 1) ) * 0.5;
                }
            }
            
#else
            } else {
                tmp = (_w->at(ind) + _w->at(ind + 1)) * 0.5;
                tmp1 = (_w1->at(ind) + _w1->at(ind + 1) ) * 0.5;
            }
#endif
                    
            k0->at(x) = tmp / t8xi3;
            k1->at(x) = (-tmp1-3*tmp)/ t8xi3;
        }
    }
};

#ifdef LIBKEDF_OCL
#include "HuangCarterOCL.hpp"
#endif

#endif /* HUANGCARTER_HPP */

