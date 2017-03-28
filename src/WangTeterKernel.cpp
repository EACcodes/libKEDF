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
#include <cmath>
#include "HelperFunctions.hpp"
#include "WangTeter.hpp"

WangTeterKernel::WangTeterKernel(const double alpha, const double beta, const double rho0, const double lambdaTF, const double muVW, const double ft){
    _alpha = alpha;
    _beta = beta;
    _rho0 = rho0;
    _ft = ft;
    _lambda = lambdaTF;
    _mu = muVW;

    _coeff = 5.0/(9.0*_alpha*_beta*pow(_rho0,_alpha+_beta-_ft));
    const double inner = 3.0*rho0*M_PI*M_PI;
    const double inner13 = std::cbrt(inner);
    _tkF = 2.0*inner13;
}

WangTeterKernel::~WangTeterKernel(){
}

void WangTeterKernel::fillWTKernelReciprocal(cube* kernel, const cube* gNorms){

    const uword elems = kernel->n_elem;
    
    #pragma omp parallel for default(none) shared(kernel,gNorms)
    for(uword x = 0; x < elems; ++x){
        const double gNorm = gNorms->at(x);
        const double lind = MathFunctions::lindhardResponse(gNorm/_tkF,_lambda,_mu);
        kernel->at(x) = lind*_coeff;
    }
}
