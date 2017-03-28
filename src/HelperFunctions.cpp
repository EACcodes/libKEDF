/* 
 * Copyright (c) 2015-2017, Princeton University, Johannes M Dieterich, Emily A Carter
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
#include <cmath>
#include "HelperFunctions.hpp"
using namespace arma;

const static double LOWDENSITYFORCUTOFF = 1E-2;
const static double HIGHDENSITYORCUTOFF = 100.0;
const static double HIPOWER = exp(100.0);

template<class GridComputer, template<class> class GridType>
void CutoffFunctions::WGCVacuumCutoff(unique_ptr<GridType<GridComputer> >& density, const double rhoV, const double rhoStep){
    
    const double ct = exp(rhoV/rhoStep);
    const double ctP1Sq = (ct+1.0)*(ct+1.0);
    const double rhoStepSq = rhoStep*rhoStep;
    const double t1 = (rhoStep+ct*rhoStep);
    const double t2 = (2.0*ctP1Sq*rhoStepSq);
    const double cutHi = (HIPOWER-1.0)/(HIPOWER+ct);
    
    cube* dens = density->getRealGrid();
    const size_t nSlices = dens->n_slices;
    const size_t nRows = dens->n_rows;
    const size_t nCols = dens->n_cols;
    #pragma omp parallel for default(none) shared(dens)
    for(size_t x = 0; x < nSlices; ++x){
        for(size_t col = 0; col < nCols; ++col){
            for(size_t row = 0; row < nRows; ++row){
                const double div = dens->at(row,col,x)/rhoStep;
                if(div < LOWDENSITYFORCUTOFF){
                    // the cutoff function is taylor expanded to 2nd order
                    const double d = dens->at(row,col,x);
                    dens->at(row,col,x) = d/t1 + (-1.0+ct)*d*d/t2;
                } else if(div > HIGHDENSITYORCUTOFF){
                    dens->at(row,col,x) = cutHi; // density is much bigger
                } else {
                    const double powDens = exp(div);
                    dens->at(row,col,x) = (powDens-1.0)/(powDens+ct);
                }
            }
        }
    }
    density->complete(dens);
}

double CutoffFunctions::vacuumCutoff(const double density, const double rhoV, const double rhoStep){
    
    // XXX not happy about this block, as it is a lot of recomputing but to save memory, this function may still be useful
    const double ct = exp(rhoV/rhoStep);
    const double ctP1Sq = (ct+1.0)*(ct+1.0);
    const double rhoStepSq = rhoStep*rhoStep;
    const double t1 = (rhoStep+ct*rhoStep);
    const double t2 = (2.0*ctP1Sq*rhoStepSq);
    const double cutHi = (HIPOWER-1.0)/(HIPOWER+ct);
    
    const double div = density/rhoStep;
    if(div < LOWDENSITYFORCUTOFF){
        // the cutoff function is taylor expanded to 2nd order
        return density/t1 + (-1.0+ct)*density*density/t2;
    } else if(div > HIGHDENSITYORCUTOFF){
        return cutHi; // density is much bigger
    } else {
        const double powDens = exp(div);
        return (powDens-1.0)/(powDens+ct);
    }
}

template<class GridComputer, template<class> class GridType>
void CutoffFunctions::WGCVacuumCutoffDeriv(unique_ptr<GridType<GridComputer> >& density, const double rhoV, const double rhoStep){
    
    const double ct = exp(rhoV/rhoStep);
    const double ctP1Sq = (ct+1.0)*(ct+1.0);
    const double rhoStepSq = rhoStep*rhoStep;
    const double hiCtSq = (HIPOWER+ct)*(HIPOWER+ct);
    const double t1 = (rhoStep+ct*rhoStep);
    const double t2 = (ctP1Sq*rhoStepSq);
    const double cutHi = HIPOWER*(1.0+ct)/(hiCtSq*rhoStep);
    
    cube* dens = density->getRealGrid();
    const size_t nSlices = dens->n_slices;
    const size_t nRows = dens->n_rows;
    const size_t nCols = dens->n_cols;
    #pragma omp parallel for default(none) shared(dens)
    for(size_t x = 0; x < nSlices; ++x){
        for(size_t col = 0; col < nCols; ++col){
            for(size_t row = 0; row < nRows; ++row){
                const double div = dens->at(row,col,x)/rhoStep;
                if(div < LOWDENSITYFORCUTOFF){
                    // the cutoff function is taylor expanded to 2nd order
                    const double d = dens->at(row,col,x);
                    dens->at(row,col,x) = 1.0/t1 + (-1.0+ct)*d/t2;
                } else if(div > HIGHDENSITYORCUTOFF){
                    dens->at(row,col,x) = cutHi; // density is much bigger
                } else {
                    const double powDens = exp(div);
                    const double powCt = powDens+ct;
                    const double powCtSq = powCt*powCt;
                    dens->at(row,col,x) = powDens*(1.0+ct)/(powCtSq*rhoStep);
                }
            }
        }
    }
}

double CutoffFunctions::vacuumCutoffDeriv(const double density, const double rhoV, const double rhoStep){
    
    // XXX not happy about this block, as it is a lot of recomputing but to save memory, this function may still be useful
    const double ct = exp(rhoV/rhoStep);
    const double ctP1Sq = (ct+1.0)*(ct+1.0);
    const double rhoStepSq = rhoStep*rhoStep;
    const double hiCtSq = (HIPOWER+ct)*(HIPOWER+ct);
    const double t1 = (rhoStep+ct*rhoStep);
    const double t2 = (ctP1Sq*rhoStepSq);
    const double cutHi = HIPOWER*(1.0+ct)/(hiCtSq*rhoStep);
    
    const double div = density/rhoStep;
    if(div < LOWDENSITYFORCUTOFF){
        // the cutoff function is taylor expanded to 2nd order
        return 1.0/t1 + (-1.0+ct)*density/t2;
    } else if(div > HIGHDENSITYORCUTOFF){
        return  cutHi; // density is much bigger
    } else {
        const double powDens = exp(div);
        const double powCt = powDens+ct;
        const double powCtSq = powCt*powCt;
        return powDens*(1.0+ct)/(powCtSq*rhoStep);
    }
}

double MathFunctions::lindhardResponse(const double eta, const double lambda, const double mu){
    
    if(eta < 0.0){return 0.0;}
    // limit for small eta
    else if(eta < 1e-10){
        const double lind = 1.0 - lambda + eta*eta * (1.0/3.0-3.0*mu);
        return lind;
    } else if(abs(eta-1.0) < 1e-10){
        const double lind = 2.0 - lambda - 3.0*mu + 20.0*(eta-1.0);
        return lind;
    } else if(eta > 3.65){
        // Taylor expansion for high eta
        const double etaSq = eta*eta;
        const double invEtaSq = 1.0/etaSq;
        const double lind = 3.0*(1.0-mu)*etaSq
            - lambda - 0.6
            + invEtaSq *  (-0.13714285714285712
            + invEtaSq * (-6.39999999999999875E-2
            + invEtaSq * (-3.77825602968460128E-2
            + invEtaSq * (-2.51824061652633074E-2
            + invEtaSq * (-1.80879839616166146E-2
            + invEtaSq * (-1.36715733124818332E-2
            + invEtaSq * (-1.07236045520990083E-2
            + invEtaSq * (-8.65192783339199453E-3 
            + invEtaSq * (-7.1372762502456763E-3 
            + invEtaSq * (-5.9945117538835746E-3 
            + invEtaSq * (-5.10997527675418131E-3 
            + invEtaSq * (-4.41060829979912465E-3 
            + invEtaSq * (-3.84763737842981233E-3 
            + invEtaSq * (-3.38745061493813488E-3 
            + invEtaSq * (-3.00624946457977689E-3)))))))))))))));
        return lind;
    } else {
        const double lind = 1.0 / (0.5 + 0.25 * (1.-eta*eta) * log((1.0 + eta)
            / abs(1.0-eta))/eta) - 3.0 * mu * eta*eta - lambda;
        return lind;
    }
}

double MathFunctions::derivativeLindhardResponse(const double eta, const double mu){
    
    if(eta < 0.0){
        return 0.0;
    } else if(eta < 1e-10){
        return 2*eta*(1.0/3.0 - 3.0*mu);
    } else if(abs(eta-1.0) < 1e-10){
        return 40.0;
    } else {
        const double etaSq = eta*eta;
        const double oneMEta = 1-eta;
        const double onePEta = 1+eta;
        const double denom = (0.5 + 0.25*(1-eta*eta) * log(onePEta/abs(oneMEta))/eta);
        const double denomSq = denom*denom;
        const double gprim = ((etaSq + 1)*0.25 / etaSq * log(abs((1.0+eta)/(oneMEta))) - 0.5/eta) / denomSq - 6*eta*mu;
        
        return gprim;
    }
}