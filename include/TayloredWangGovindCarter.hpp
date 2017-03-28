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

#ifndef TAYLOREDWANGGOVINDCARTER_HPP
#define TAYLOREDWANGGOVINDCARTER_HPP

#include <cmath>
#include "FourierGrid.hpp"
#include "HelperFunctions.hpp"
#include "WangGovindCarter.hpp"
using namespace arma;
using namespace std;

template<class GridType>
class TayloredWangGovindCarter: public KEDF<GridType> {

public:
        
    TayloredWangGovindCarter(GridType* example, const double alpha, const double beta, const double gamma, const double rhoS, cube* kernel0th,
            cube* kernel1st) : _alpha(alpha), _beta(beta), _gamma(gamma), _rhoS(rhoS), _sndOrder(false), _vacCutoff(false),
                    _kernel0th(kernel0th), _kernel1st(kernel1st), _kernel2nd(NULL), _kernel3rd(NULL) {
        // XXX make vacCutoff configurable
    }

    TayloredWangGovindCarter(GridType* example, const double alpha, const double beta, const double gamma, const double rhoS, cube* kernel0th,
            cube* kernel1st, cube* kernel2nd, cube* kernel3rd)
            : _alpha(alpha), _beta(beta), _gamma(gamma), _rhoS(rhoS), _sndOrder(true), _vacCutoff(false),
            _kernel0th(kernel0th), _kernel1st(kernel1st), _kernel2nd(kernel2nd), _kernel3rd(kernel3rd){
        // XXX make vacCutoff configurable
    }

    ~TayloredWangGovindCarter(){
    }

    string getMethodDescription() const {
        if(_sndOrder){
            return "2nd order Taylor-expanded Wang-Govind-Carter (1999) KEDF)" ;
        } else {
            return "1st order Taylor-expanded Wang-Govind-Carter (1999) KEDF)" ;
        }
    }

    vector<string> getCitations() const {
        vector<string> citations(0);
        citations.push_back("sorry, not yet");
    
        return citations;
    }

    vector<string> getWorkingEquations() const {
        vector<string> citations(0);
        citations.push_back("sorry, not yet");
    
        return citations;
    }

    double calcEnergy(const GridType& grid) const {
    
        if(!_sndOrder && !_vacCutoff){
            // first order Taylor expansion, no vacuum cutoff employed
        
            unique_ptr<FourierGrid> workGridA = grid.createFourierDuplicate();
            unique_ptr<FourierGrid> scr1 = grid.createFourierEmptyDuplicate();
            unique_ptr<FourierGrid> scr2 = grid.createFourierEmptyDuplicate();
        
            const cube* density = grid.tryReadRealGrid();
            if(!density){
                // unfortunately, we need to get a copy of the grid
                unique_ptr<FourierGrid> workGrid = grid.createFourierDuplicate();
                density = workGrid->readRealGrid();
            }
        
            cube* densityA = workGridA->getRealGrid();
            cube* scr1Dens = scr1->getRealGrid();

            const uword elems = densityA->n_elem;
        
            #pragma omp parallel for default(none) shared(densityA,scr1Dens)
            for(uword x = 0; x < elems; ++x){
                
                // store density**alpha and first order expansion in real space multiplied by rho**alpha
                const double dens = densityA->at(x);
                const double densAlpha = pow(dens,_alpha);
                densityA->at(x) = densAlpha;
                scr1Dens->at(x) = (dens - 2*_rhoS) * densAlpha; 
            }
        
            workGridA->complete(densityA);
            scr1->complete(scr1Dens);
            
            // move densA and scr1 into g space, scr2 is whatever
            const cx_cube* recA = workGridA->readReciprocalGrid();
            cx_cube* recScr1 = scr1->getReciprocalGrid();
            cx_cube* recScr2 = scr2->getReciprocalGrid();
        
            const uword recElems = recA->n_elem;
    
            #pragma omp parallel for default(none) shared(recA,recScr1,recScr2)
            for(uword x = 0; x < recElems; ++x){
                    
                const complex<double> curr = recScr1->at(x);
                recScr1->at(x) = _kernel0th->at(x)*recA->at(x)
                        + _kernel1st->at(x)*curr;
                
                recScr2->at(x) = _kernel1st->at(x)*recA->at(x);
            }
            scr1->completeReciprocal(recScr1);
            scr2->completeReciprocal(recScr2);
            workGridA.reset();
        
            // get the scr1 and scr2 into real space, we only need one of them to be writable
            const cube* scr1Dens_2 = scr1->readRealGrid();
            cube* scr2Dens_2 = scr2->getRealGrid();
            
            #pragma omp parallel for default(none) shared(density,scr1Dens_2,scr2Dens_2)
            for(uword x = 0; x < elems; ++x){
                    
                const double dens = density->at(x);
                const double densB = pow(dens,_beta);
                const double s1 = scr1Dens_2->at(x);
                const double s2 = scr2Dens_2->at(x);
                
                scr2Dens_2->at(x) = densB*(s1 + dens*s2);
            }
            scr2->complete(scr2Dens_2);
        
            scr1.reset();
            
            const double energy = _CTF*scr2->integrate();
            
            scr2.reset();
        
            return energy;
        } else if(_sndOrder && !_vacCutoff){
        
            // second order Tayler expansion, no vacuum cutoff
        
            unique_ptr<FourierGrid> workGridA = grid.createFourierDuplicate();
            unique_ptr<FourierGrid> scr1 = grid.createFourierEmptyDuplicate();
            unique_ptr<FourierGrid> scr2 = grid.createFourierEmptyDuplicate();
        
            const cube* density = grid.tryReadRealGrid();
            if(!density){
                // unfortunately, we need to get a copy of the grid
                unique_ptr<FourierGrid> workGrid = grid.createFourierDuplicate();
                density = workGrid->readRealGrid();
            }
        
            cube* densityA = workGridA->getRealGrid();
            cube* scr1Dens = scr1->getRealGrid();
            cube* scr2Dens = scr2->getRealGrid();
        
            const uword elems = densityA->n_elem;
            const size_t nSlices = densityA->n_slices;
            const size_t nRows = densityA->n_rows;
            const size_t nCols = densityA->n_cols;
        
            auto theta = MemoryFunctions::allocateScratch(nRows, nCols, nSlices);
        
            #pragma omp parallel for default(none) shared(densityA,scr1Dens,scr2Dens,theta)
            for(uword x = 0; x < elems; ++x){
                
                // theta and dens**alpha are reused a lot
                const double dens = densityA->at(x);
                const double densAlpha = pow(dens,_alpha);
                densityA->at(x) = densAlpha;
                const double th = dens - _rhoS;
                theta->at(x) = th;
                scr1Dens->at(x) = th * densAlpha;
                scr2Dens->at(x) = th*th * densAlpha * 0.5;
            }
            workGridA->complete(densityA);
            scr2->complete(scr2Dens);
            scr1->complete(scr1Dens);
        
            // move densA, scr1, and scr2 into g space
            const cx_cube* recA = workGridA->readReciprocalGrid();
            cx_cube* recScr1 = scr1->getReciprocalGrid();
            cx_cube* recScr2 = scr2->getReciprocalGrid();
        
            unique_ptr<FourierGrid> scr3 = grid.createFourierEmptyDuplicate();
            cx_cube* recScr3 = scr3->getReciprocalGrid();
        
            const uword recElems = recA->n_elem;
    
            #pragma omp parallel for default(none) shared(recA,recScr1,recScr2,recScr3)
            for(uword x = 0; x < recElems; ++x){
                    
                const complex<double> curr1 = recScr1->at(x);
                const complex<double> curr2 = recScr2->at(x);
                
                recScr1->at(x) = _kernel0th->at(x)*recA->at(x)
                        + _kernel1st->at(x)*curr1
                        + _kernel2nd->at(x)*curr2;
                
                recScr2->at(x) = _kernel1st->at(x)*recA->at(x)
                        + _kernel3rd->at(x)*curr1;
                
                recScr3->at(x) = _kernel2nd->at(x)*recA->at(x);
            }
            scr1->completeReciprocal(recScr1);
            scr2->completeReciprocal(recScr2);
            scr3->completeReciprocal(recScr3);
        
        
            // we can delete densityA and instead allocate densityB, to avoid any background FFT
            workGridA.reset();
            unique_ptr<FourierGrid> scr4 = grid.createFourierEmptyDuplicate();
            cube* scr4Dens = scr4->getRealGrid();
        
            // get the scr1 and scr2 into real space, we only need them to be readable
            const cube* scr1Dens_2 = scr1->readRealGrid();
            const cube* scr2Dens_2 = scr2->readRealGrid();
            const cube* scr3Dens_2 = scr3->readRealGrid();
        
            #pragma omp parallel for default(none) shared(density,densityA,scr4Dens,theta,scr1Dens_2,scr2Dens_2,scr3Dens_2)
            for(uword x = 0; x < elems; ++x){
                const double densB = pow(density->at(x),_beta);
                const double th = theta->at(x);
                scr4Dens->at(x) = densB*(scr1Dens_2->at(x) + th * scr2Dens_2->at(x)
                        + th*th*0.5*scr3Dens_2->at(x));
            }
            scr4->complete(scr4Dens);
        
            scr1.reset();
            scr2.reset();
            scr3.reset();
            
            const double energy = _CTF*scr4->integrate();
            
            scr4.reset();
            
            return energy;
        }
    
        throw runtime_error("XXX implement me");
    }

    double calcPotential(const GridType& grid, GridType& potential) const {
    
        if(!_sndOrder && !_vacCutoff){
            // first order Taylor expansion, no vacuum cutoff employed
        
            unique_ptr<FourierGrid> workGridA = grid.createFourierDuplicate();
            unique_ptr<FourierGrid> workGridB = grid.createFourierEmptyDuplicate();
            unique_ptr<FourierGrid> scr1 = grid.createFourierEmptyDuplicate();
            unique_ptr<FourierGrid> scr2 = grid.createFourierEmptyDuplicate();
        
            const cube* density = grid.tryReadRealGrid();
            if(!density){
                // unfortunately, we need to get a copy of the grid
                unique_ptr<FourierGrid> workGrid = grid.createFourierDuplicate();
                density = workGrid->readRealGrid();
            }
        
            cube* densityA = workGridA->getRealGrid();
            cube* densityB = workGridB->getRealGrid();
            cube* scr1Dens = scr1->getRealGrid();
        
            const uword elems = densityA->n_elem;
        
            #pragma omp parallel for default(none) shared(densityA,scr1Dens)
            for(uword x = 0; x < elems; ++x){
                
                // store density**alpha and first order expansion in real space multiplied by rho**alpha
                const double dens = densityA->at(x);
                const double densAlpha = pow(dens,_alpha);
                densityA->at(x) = densAlpha;
                scr1Dens->at(x) = (dens - 2*_rhoS) * densAlpha; 
            }
        
            workGridA->complete(densityA);
        
            scr1->complete(scr1Dens);
        
            // move densA and scr1 into g space, scr2 is whatever
            const cx_cube* recA = workGridA->readReciprocalGrid();
            cx_cube* recScr1 = scr1->getReciprocalGrid();
            cx_cube* recScr2 = scr2->getReciprocalGrid();
        
            const uword recElems = recA->n_elem;
    
            #pragma omp parallel for default(none) shared(recA,recScr1,recScr2)
            for(uword x = 0; x < recElems; ++x){
                    
                const complex<double> curr = recScr1->at(x);
                recScr1->at(x) = _kernel0th->at(x)*recA->at(x)
                        + _kernel1st->at(x)*curr;
                
                recScr2->at(x) = _kernel1st->at(x)*recA->at(x);
            }
            scr1->completeReciprocal(recScr1);
            scr2->completeReciprocal(recScr2);
        
            // get the scr1 and scr2 into real space, we only need them to be readable
            cube* scr1Dens_2 = scr1->getRealGrid();
            const cube* scr2Dens_2 = scr2->readRealGrid();
        
            // we can delete densityA and instead allocate scr3, to avoid any background FFT
            workGridA.reset();
            unique_ptr<FourierGrid> scr3 = grid.createFourierEmptyDuplicate();
            cube* scr3Dens = scr3->getRealGrid();
            cube* pot = potential.getRealGrid();
        
            #pragma omp parallel for default(none) shared(pot,density,densityB,scr1Dens_2,scr2Dens_2,scr3Dens)
            for(uword x = 0; x < elems; ++x){
                    
                const double dens = density->at(x);
                const double densBM1 = pow(dens,_beta-1);
                const double densB = densBM1*dens;
                densityB->at(x) = densB;
                const double s1 = scr1Dens_2->at(x);
                const double s2 = scr2Dens_2->at(x);
                
                const double s3 = densB*(s1 + dens*s2);
                scr3Dens->at(x) = s3;
                
                pot->at(x) = _beta*densBM1 * s1 + (_beta+1) * densB*s2;
                scr1Dens_2->at(x) = (dens - 2*_rhoS) * densB; 
            }
            scr3->complete(scr3Dens);
            workGridB->complete(densityB);
            scr1->complete(scr1Dens_2);
            
            const double energy = _CTF*scr3->integrate();
            
            scr3.reset(); // we can reset this now, doesn't probably make a big difference, but is cleaner
        
            // calculate the potential w.r.t. density**alpha
        
            const cx_cube* recB = workGridB->readReciprocalGrid();
            cx_cube* recScr1_2 = scr1->getReciprocalGrid();
        
            scr2->resetToReciprocal(); // to avoid background FFT
            cx_cube* recScr2_2 = scr2->getReciprocalGrid();
        
            #pragma omp parallel for default(none) shared(recB,recScr1_2,recScr2_2)
            for(uword x = 0; x < recElems; ++x){
                
                const complex<double> curr = recScr1_2->at(x);
                recScr1_2->at(x) = _kernel0th->at(x)*recB->at(x)
                        + _kernel1st->at(x)*curr;
                
                recScr2_2->at(x) = _kernel1st->at(x)*recB->at(x);
            }
            scr2->completeReciprocal(recScr2_2);
            scr1->completeReciprocal(recScr1_2);
            
            workGridB.reset();
        
            const cube* scr1Dens_3 = scr1->readRealGrid();
            const cube* scr2Dens_3 = scr2->readRealGrid();
        
            #pragma omp parallel for default(none) shared(pot,density,scr1Dens_3,scr2Dens_3)
            for(uword x = 0; x < elems; ++x){
                    
                const double dens = density->at(x);
                const double s1 = scr1Dens_3->at(x);
                const double s2 = scr2Dens_3->at(x);
                const double densAM1 = pow(dens,_alpha-1);
                const double densAlpha = densAM1*dens;
                    
                pot->at(x) += _alpha*densAM1 * s1 + (_alpha+1) * densAlpha*s2; // XXX should we cache dens**alpha?
                pot->at(x) *= _CTF;
            }
        
            scr1.reset();
            scr2.reset();
            
            potential.complete(pot);
        
            return energy;
        } else if(_sndOrder && !_vacCutoff){
        
            // second order Tayler expansion, no vacuum cutoff
        
            unique_ptr<FourierGrid> workGridA = grid.createFourierDuplicate();
            unique_ptr<FourierGrid> scr1 = grid.createFourierEmptyDuplicate();
            unique_ptr<FourierGrid> scr2 = grid.createFourierEmptyDuplicate();
        
            const cube* density = grid.tryReadRealGrid();
            if(!density){
                // unfortunately, we need to get a copy of the grid
                unique_ptr<FourierGrid> workGrid = grid.createFourierDuplicate();
                density = workGrid->readRealGrid();
            }
        
            cube* densityA = workGridA->getRealGrid();
            cube* scr1Dens = scr1->getRealGrid();
            cube* scr2Dens = scr2->getRealGrid();
        
            const uword elems = densityA->n_elem;
            const size_t nSlices = densityA->n_slices;
            const size_t nRows = densityA->n_rows;
            const size_t nCols = densityA->n_cols;
        
            auto theta = MemoryFunctions::allocateScratch(nRows, nCols, nSlices);
        
            #pragma omp parallel for default(none) shared(densityA,scr1Dens,scr2Dens,theta)
            for(uword x = 0; x < elems; ++x){
                
                // theta and dens**alpha are reused a lot
                const double dens = densityA->at(x);
                const double densAlpha = pow(dens,_alpha);
                densityA->at(x) = densAlpha;
                const double th = dens - _rhoS;
                theta->at(x) = th;
                scr1Dens->at(x) = th * densAlpha;
                scr2Dens->at(x) = th*th * densAlpha * 0.5;
            }
            workGridA->complete(densityA);
            scr2->complete(scr2Dens);
            scr1->complete(scr1Dens);
        
            // move densA, scr1, and scr2 into g space
            const cx_cube* recA = workGridA->readReciprocalGrid();
            cx_cube* recScr1 = scr1->getReciprocalGrid();
            cx_cube* recScr2 = scr2->getReciprocalGrid();
        
            unique_ptr<FourierGrid> scr3 = grid.createFourierEmptyDuplicate();
            cx_cube* recScr3 = scr3->getReciprocalGrid();
        
            const uword recElems = recA->n_elem;
    
            #pragma omp parallel for default(none) shared(recA,recScr1,recScr2,recScr3)
            for(uword x = 0; x < recElems; ++x){
                    
                const complex<double> curr1 = recScr1->at(x);
                const complex<double> curr2 = recScr2->at(x);
                
                recScr1->at(x) = _kernel0th->at(x)*recA->at(x)
                        + _kernel1st->at(x)*curr1
                        + _kernel2nd->at(x)*curr2;
                
                recScr2->at(x) = _kernel1st->at(x)*recA->at(x)
                        + _kernel3rd->at(x)*curr1;
                
                recScr3->at(x) = _kernel2nd->at(x)*recA->at(x);
            }
            scr1->completeReciprocal(recScr1);
            scr2->completeReciprocal(recScr2);
            scr3->completeReciprocal(recScr3);
        
        
            // we can delete densityA and instead allocate densityB, to avoid any background FFT
            workGridA.reset();
            unique_ptr<FourierGrid> workGridB = grid.createFourierEmptyDuplicate();
            cube* densityB = workGridB->getRealGrid();
        
            // get the scr1, scr2, and scr3 into real space, we only need them to be readable
            cube* scr1Dens_2 = scr1->getRealGrid();
            cube* scr2Dens_2 = scr2->getRealGrid();
            cube* scr3Dens_2 = scr3->getRealGrid();
        
            // setup grid for energy integration and first half of the potential
            cube* pot = potential.getRealGrid();
            #pragma omp parallel for default(none) shared(density,densityB,scr3Dens_2,pot,theta,scr1Dens_2,scr2Dens_2)
            for(uword x = 0; x < elems; ++x){
                
                const double dens = density->at(x);
                const double th = theta->at(x);
                const double s1 = scr1Dens_2->at(x);
                const double s2 = scr2Dens_2->at(x);
                const double s3 = scr3Dens_2->at(x);

                const double densBM1 = pow(dens,_beta-1);
                const double densB = densBM1*dens;
                
                densityB->at(x) = densB;                    
                scr3Dens_2->at(x) = densB*(s1 + th*s2 + th*th*0.5*s3);
                
                pot->at(x) = densBM1 * (_beta*s1
                        + (dens + _beta*th) * s2
                        + th*(dens+_beta*th*0.5)*s3);
                
                // now reuse the scr1/scr2
                scr1Dens_2->at(x) = th*densB;
                scr2Dens_2->at(x) = th*th * densB * 0.5;
            }
            workGridB->complete(densityB);
            scr3->complete(scr3Dens_2);
            scr2->complete(scr2Dens_2);
            scr1->complete(scr1Dens_2);
        
            const double energy = _CTF*scr3->integrate();
        
            // get density**beta and the th*densB grid to the reciprocal space
            const cx_cube* recB = workGridB->readReciprocalGrid();
            cx_cube* recScr1_2 = scr1->getReciprocalGrid(); // contains theta*densB
            cx_cube* recScr2_2 = scr2->getReciprocalGrid(); // contains theta*theta*densB*0.5
        
            // scr3 can be reused, we need it in the reciprocal space
            scr3->resetToReciprocal(); // to avoid background FFT
            cx_cube* recScr3_2 = scr3->getReciprocalGrid();
        
            #pragma omp parallel for default(none) shared(recB,recScr1_2,recScr2_2,recScr3_2)
            for(uword x = 0; x < recElems; ++x){
                    
                const complex<double> rB = recB->at(x);
                const complex<double> s1 = recScr1_2->at(x);
                const complex<double> s2 = recScr2_2->at(x);
                recScr3_2->at(x) = _kernel0th->at(x) * rB
                        + _kernel1st->at(x) * s1
                        + _kernel2nd->at(x) * s2;

                recScr1_2->at(x) = _kernel1st->at(x) * rB
                        + _kernel3rd->at(x) * s1;
                
                recScr2_2->at(x) = _kernel2nd->at(x) * rB;
            }
            scr3->completeReciprocal(recScr3_2);
            scr2->completeReciprocal(recScr2_2);
            scr1->completeReciprocal(recScr1_2);
        
            // get them over to real space again
            const cube* scr3Dens_3 = scr3->readRealGrid(); // 0th-order kernel * density**beta + 1st-order kernel * [theta*densB] + 2nd-order kernel * [theta*theta*densB*0.5]
            const cube* scr1Dens_3 = scr1->readRealGrid(); // 1st-order kernel * density**beta + 3rd-order kernel * [theta*densB]
            const cube* scr2Dens_3 = scr2->readRealGrid(); // 2nd-order kernel * density**beta
        
            // assemble the second half of the potential
        
            #pragma omp parallel for default(none) shared(density,scr1Dens_3,scr2Dens_3,scr3Dens_3,theta,pot)
            for(uword x = 0; x < elems; ++x){
                    
                const double dens = density->at(x);
                const double densAM1 = pow(dens,_alpha-1);
                const double s1 = scr1Dens_3->at(x);
                const double s2 = scr2Dens_3->at(x);
                const double s3 = scr3Dens_3->at(x);
                const double th = theta->at(x);
                    
                pot->at(x) += densAM1 * (_alpha*s3 + (dens+_alpha*th)*s1
                        + th*(dens+_alpha*th*0.5)*s2);
                pot->at(x) *= _CTF;
            }
            
            scr1.reset();
            scr2.reset();
            scr3.reset();
        
            potential.complete(pot);
        
            return energy;
        }
    
        throw runtime_error("XXX implement me");
    }

    unique_ptr<StressTensor> calcStress(const GridType& grid) const {
        throw runtime_error("XXX implement me");
    }

    
private:
    const double _CTF = 2.87123400018819;
    const double _alpha;
    const double _beta;
    const double _gamma;
    const double _rhoS;
    const bool _sndOrder;
    const bool _vacCutoff;
    const cube * _kernel0th;
    const cube * _kernel1st;
    const cube * _kernel2nd;
    const cube * _kernel3rd;
};

#ifdef LIBKEDF_OCL
#include "TayloredWangGovindCarterOCL.hpp"
#endif

#endif /* TAYLOREDWANGGOVINDCARTER_HPP */

