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
#include <cmath>
#include <spline.hpp>
#include "WangGovindCarter.hpp"
#include "IntKernelODE.hpp"
using namespace arma;

WangGovindCarterODE::WangGovindCarterODE(const double beta, const double gamma){
    _beta = beta;
    _gamma = gamma;
}

WangGovindCarterODE::~WangGovindCarterODE(){
}

/**
 * here is your kernel ODE, you need to give y, and y' upon given t
 * i.e. here y is just the w(eta) and t is just the eta
 * this NLS_TF ODE is the equation (26) in David Garcia-Aldea and 
 * J.E. Alvarellos's  paper (PRA)2007
 */
void WangGovindCarterODE::evaluate(const double eta, double y[], double yp[]) {
    
    yp[0] = y[1];
    
    // check eta
    if(eta < 0.0){
        throw runtime_error("WGC ODE: eta < 0.0.");
    }
    
    double fLind = -42.0;
    if(eta == 1.0){
        fLind = 2.0;
    } else if(eta == 0.0){
        fLind = 1.0;
    } else {
        fLind = 1.0/(0.5+(1-eta*eta)/(4*eta)*log(abs((1+eta)/(1-eta))));
    }
        
    //  WGC formula, Alex Wang, Govind, Carter, PRB 1999
    const double wgc = 20 * (fLind- 3*eta*eta -1) -
             ( _gamma+1.0-6.0*5.0/3.0)*eta*y[1] -
             36.0 * (5.0/3.0-_beta) * _beta * y[0];
    
    yp[1] = wgc/(eta*eta);
    
    //cout.precision(11);
    //cout << "KERNEL1 " << _gamma << "\t" << _beta << "\t" << y[0] << "\t" << y[1] << endl;
    //cout << "KERNEL: " << eta << "\t" << fLind << "\t" << yp[1] << endl;
}

FullWangGovindCarterODE::FullWangGovindCarterODE(const double beta, const double gamma){
    _beta = beta;
    _gamma = gamma;
}

FullWangGovindCarterODE::~FullWangGovindCarterODE(){
}

/**
 * here is your kernel ODE, you need to give y, and y' upon given t
 * i.e. here y is just the w(eta) and t is just the eta
 * this NLS_TF ODE is the equation (26) in David Garcia-Aldea and 
 * J.E. Alvarellos's  paper (PRA)2007
 */
double FullWangGovindCarterODE::evaluate(const double eta, vec *y, size_t m) {
    
    const double etaP1 = eta+1;
    const double oneMEta = 1-eta;
    const double oneMEtaSq = 1-eta*eta;
    const double twoEta = 2*eta;
    const double etaSq = eta*eta;
    const double fac1 = oneMEtaSq/twoEta;
    
    double fac2 = 0.0;
    if(eta != 1.0){
        fac2 = log(abs(etaP1/oneMEta));
    }
    const double fLind = 1+fac1*fac2;
    
    double res = -42.0;
    if(m == 0){
        res = y->at(1);
    } else if(m == 1){
        const double fTilde = 2.0/fLind - 3*etaSq + 3.0/5.0;
        res = (125.0/32.0)*fTilde+ eta*y->at(1) - (25.0/4.0)*y->at(0);
        res /= etaSq;
    }
    
    return res;
}

 /*
  * THIS CODE IS A TRANSLATION OF THE PROFESS2 ROUTINE BASED ON Ho/Ligneres/Carter's
  * ANALYTICAL WGC KERNEL PAPER
  */
AnalyticalWangGovindCarterKernel::AnalyticalWangGovindCarterKernel(const double alpha, const double beta, const double gamma, const size_t numTermsAB){
    
    if(true){
        throw runtime_error("implementation not complete");
    }
    _numTermsAB = numTermsAB;
    _pea = 3 * (alpha+beta) - gamma/2;
    _an = vector<double>(_numTermsAB+1);
    _bn = vector<double>(_numTermsAB+1);
    // XXX all problematic
    _cue = -100000;
    _ell = -100000;
    _cOne = -100000;
    _cTwo = -100000;
    fillAB();
}

AnalyticalWangGovindCarterKernel::~AnalyticalWangGovindCarterKernel(){
}

void AnalyticalWangGovindCarterKernel::elementWGC(const double eta, array<int,3>& w){
    
    const double etaSq = eta*eta;
    
    double w0 = 0.0, w1 = 0.0, w2 = 0.0;
    
    if(eta < 1.0 || (eta == 1.0 && _pea < 0.0)){
        // set the particular solution in term of the bn series.
        for(size_t i = 1; i <= _numTermsAB; ++i){
            const double t0 = _bn[i]*pow(etaSq,i);
            const double t1 = 2.0*i;
            const double t2 = t0*t1;
            
            w0 += t0;
            w1 += t2;
            w2 += (t2*(t1-1));
        }
    } else {
        // otherwise we need to use the an series.
        for(size_t i = 1; i <= _numTermsAB; ++i){
            const double t0 = _an[i]/pow(etaSq,i);
            const double t1 = 2.0*i;
            
            w0 += t0;
            w1 -= t0*t1;
            w2 += t0*t1*(t1+1);
        }
    }
    
    /*
     * This takes care of the particular solution. Now add the general 
     * one if needed. Here we make sure that eta is not 0 because we 
     * don't want to take the log of 0. In the case where eta=0, the 
     * general solution is 0.
     */
    if((eta-1)*_pea < 0.0 && eta > 0.0){
        
        if(_cue < 0.0){
            const double etaP = pow(eta,_pea);
            const double t1 = _ell*log(eta);
            const double c1 = cos(t1);
            const double c2 = sin(t1);
            
            w0 += etaP*(_cOne*c1+_cTwo*c2);
            
            const double t1a = _pea*_cOne + _ell*_cTwo;
            const double t2a = _pea*_cTwo - _ell*_cOne;
            
            w1 += etaP * (t1a * c1 + t2a * c2);
            const double pea_m1 = _pea-1;
            w2 += etaP*((pea_m1*t1a + _ell*t2a)*c1 +(pea_m1*t2a - _ell*t1a)*c2);
        } else if(_cue == 0.0){
            const double etaP = pow(eta,_pea);
            const double t1 = _cOne+_cTwo*log(eta);
            
            w0 += etaP*t1;

            const double t1a = _pea*t1+_cTwo;
            
            w1 += etaP*t1a;
            w2 += etaP*((_pea-1)*t1a + _cTwo*_pea);
        } else {
            const double c1 = _pea + _ell;
            const double c2 = _pea - _ell;
            const double t1 = _cOne*pow(eta,c1);
            const double t2 = _cTwo*pow(eta,c2);
            
            w0 += t1+t2;
            
            const double t1a = t1*c1;
            const double t2a = t2*c2;
            
            w1 += t1a*t2a;
            w2 += (c1-1)*t1a + (c2-1)*t2a;
        }
    }
    
    w[0] = w0;
    w[1] = w1;
    w[2] = w2;
}

void AnalyticalWangGovindCarterKernel::fillAB(){
    if(true){
        throw runtime_error("implementation not complete");
    }
    _aM1 = 3;
    
}

NumericalWangGovindCarterKernel::NumericalWangGovindCarterKernel(const double alpha, const double beta, const double gamma, const double rhoS) {
    
    const double d = (3 * M_PI*M_PI * rhoS);
    _tkFStar = 2 * pow(d,1.0/3.0);
    _alpha = alpha;
    _beta = beta;
    _gamma = gamma;
    _rhoS = rhoS;
    
    // integrate the kernel
    WangGovindCarterODE* ode = new WangGovindCarterODE(_beta, _gamma);
    const double wInf = -1.6 * 20 / (36.0*_alpha*_beta);

    IntKernelODE* odeInt =  new IntKernelODE(true); // XXX stats for the time being
    _w = odeInt->makeSecondOrderKernel(ode,wInf);
    
    _eta = odeInt->getLastEta();
    
    // feed it into the spline library
    _nVals = _eta->n_rows;
    _nls_wpp  = spline_cubic_set(_nVals, _eta.get()->memptr(), _w->colptr(0), 0, 0.0, 0, 0.0);
    _nls_w1pp = spline_cubic_set(_nVals, _eta.get()->memptr(), _w->colptr(1), 0, 0.0, 0, 0.0);
    _nls_w2pp = spline_cubic_set(_nVals, _eta.get()->memptr(), _w->colptr(2), 0, 0.0, 0, 0.0);
    
    delete ode;
    delete odeInt;
}

NumericalWangGovindCarterKernel::~NumericalWangGovindCarterKernel() {
    delete[] _nls_wpp;
    delete[] _nls_w1pp;
    delete[] _nls_w2pp;
}

void NumericalWangGovindCarterKernel::fillWGCKernel(cube* kernel0th, const cube* gNorms) {
    fillWGCKernel(kernel0th,NULL,gNorms);
}

void NumericalWangGovindCarterKernel::fillWGCKernel(cube* kernel0th, cube* kernel1st, const cube* gNorms) {
    fillWGCKernel(kernel0th,kernel1st,NULL,gNorms);
}
    
void NumericalWangGovindCarterKernel::fillWGCKernel(cube* kernel0th, cube* kernel1st, cube* kernel2nd, const cube* gNorms) {
    fillWGCKernel(kernel0th,kernel1st,kernel2nd,NULL,gNorms);
}

void NumericalWangGovindCarterKernel::fillWGCKernel(cube* kernel0th, cube* kernel1st, cube* kernel2nd, cube* kernel3rd, const cube* gNorms){
    
    const size_t nSlices = kernel0th->n_slices;
    const size_t nRows = kernel0th->n_rows;
    const size_t nCols = kernel0th->n_cols;
    
    // XXX untested if the spline library is threadsafe
    #pragma omp parallel for default(none) shared(kernel0th,kernel1st,kernel2nd,kernel3rd,gNorms)
    for(size_t x = 0; x < nSlices; ++x){
        for(size_t col = 0; col < nCols; ++col){
            for(size_t row = 0; row < nRows; ++row){
                
                const double eta = gNorms->at(row,col,x) / _tkFStar;
                
                double ypval;
                double yppval;
                const double splineVal0 = spline_cubic_val (_nVals, _eta.get()->memptr(), _w->colptr(0),
                        _nls_wpp , eta, &ypval, &yppval);
                kernel0th->at(row,col,x) = splineVal0;
                
                if(kernel1st){
                    const double splineVal1 = spline_cubic_val (_nVals, _eta.get()->memptr(), _w->colptr(1),
                            _nls_w1pp, eta, &ypval, &yppval);
                    
                    kernel1st->at(row,col,x) = -splineVal1 / (6 * _rhoS);
                    
                    if(kernel2nd){
                        const double splineVal2 = spline_cubic_val (_nVals, _eta.get()->memptr(), _w->colptr(2),
                                _nls_w2pp, eta, &ypval, &yppval);
                        
                        kernel2nd->at(row,col,x) = (splineVal2 + (7-_gamma)*splineVal1) / (36 * _rhoS*_rhoS);
                        
                        if(kernel3rd){
                            kernel3rd->at(row,col,x) = (splineVal2 + (1+_gamma)*splineVal1) / (36 * _rhoS*_rhoS);
                        }
                    }
                }
            }
        }
    }
}

NumericalFullWangGovindCarterKernel::NumericalFullWangGovindCarterKernel(const double alpha, const double beta, const double gamma, const double rhoS) {
    
    const double d = (3 * M_PI*M_PI * rhoS);
    _tkFStar = 2 * pow(d,1.0/3.0);
    _alpha = alpha;
    _beta = beta;
    _gamma = gamma;
    _rhoS = rhoS;    
}

NumericalFullWangGovindCarterKernel::~NumericalFullWangGovindCarterKernel() {
}

void NumericalFullWangGovindCarterKernel::fillWGCKernel(vec* x0, vec* w, vec* w1){
    
    const double myWInf = -0.5929319411475393;
    const int mmax = 5;
    const int nmax = 100000;
    const int n = 10000;
    const int m = 2;
    const double dr = 0.01;
    const int nr = 100;
    
    const double b = 10.0;
    
    unique_ptr<vec> x = make_unique<vec>(nmax+1);
    unique_ptr<vec> facl = make_unique<vec>(nmax+1);
    unique_ptr<vec> wofr = make_unique<vec>(nmax+1);
    unique_ptr<mat> y = make_unique<mat>(mmax,nmax+1);
    
    x->at(0) = 1.0;
    y->at(0,0) = myWInf;
    y->at(1,0) = -0.2935928266164192E+01;
    
    const double interval = (b-x->at(0))/n;
    FullWangGovindCarterODE* ode = new FullWangGovindCarterODE(_beta,_gamma);
    
    rungeKutta(ode, x.get(), y.get(), n , m, b, interval);
    
    // copy wofeta over
    for(size_t i = 0; i < n; ++i){
        x0->at(i) = x->at(i);
        w->at(i) = y->at(0,i);
        w1->at(i) = y->at(1,i);
    }
    
    // we want wofeta for the time being
    if(true){return;}
    
    unique_ptr<vec> wofr1 = make_unique<vec>(nmax+1);
    
    // get the kernel into wofr from wofeta
    double norm = 0.0;
    for(size_t j = 0; j < nr; ++j){
        const double r = dr*j;
        double total = 0.0;
        for(int i = n-1; i >= 0; --i){
            const double wofeta = y->at(0,i);
            const double eta = x->at(i);
            const double etar = eta*r;
            if(etar <= 1e-6){
                facl->at(i) = eta*eta*wofeta;
            } else {
                facl->at(i) = (sin(etar)/etar)*eta*eta*wofeta;
            }
            total += facl->at(i)*abs(interval);
        }
        wofr->at(j)=4*M_PI*total/(pow(2*M_PI,3));
        norm += wofr->at(j)*r*r*dr;
    }
    
    norm *= 4*M_PI;
        
    // copy over
    for(size_t j = 0; j < nr; ++j){
        //const double r = dr*j;
        if(j == nr){
            wofr->at(j+1) = wofr->at(j);
        }
        if(j == 0){
            // copy r, wofr->at(j)
        } else {
            wofr1->at(j)=(wofr->at(j+1)-wofr->at(j-1))/dr;
            // copy r, wofr->at(j), wofr1->at(j));
        }
    }
}

void NumericalFullWangGovindCarterKernel::rungeKutta(FullWangGovindCarterODE* ode, vec *x, mat *y, size_t n, size_t m, double b, double h){
    
    const int stepmax = 4;
    unique_ptr<vec> v = make_unique<vec>(m);
    unique_ptr<mat> t = make_unique<mat>(m,stepmax);
    
    for (size_t i = 1; i < n+1; ++i){
        
        x->at(i) = x->at(i-1)+h;
        double u = x->at(i-1);
        
        for(size_t j = 0; j < m; ++j){
            v->at(j) = y->at(j,i-1);
        }
        for(size_t j = 0; j < m; ++j){
            t->at(j,0) = ode->evaluate(u,v.get(),j);
        }
        u += 0.5*h;
        for(size_t j = 0; j < m; ++j){
            v->at(j) += 0.5*h*t->at(j,0);
        }
        for(size_t j = 0; j < m; ++j){
            t->at(j,1) = ode->evaluate(u,v.get(),j);
        }
        for(size_t j = 0; j < m; ++j){
            v->at(j) = y->at(j,i-1) + 0.5*h*t->at(j,1);
        }
        for(size_t j = 0; j < m; ++j){
            t->at(j,2) = ode->evaluate(u,v.get(),j);
        }
        u += 0.5*h;
        for(size_t j = 0; j < m; ++j){
            v->at(j) = y->at(j,i-1) + h*t->at(j,2);
        }
        for(size_t j = 0; j < m; ++j){
            t->at(j,3) = ode->evaluate(u,v.get(),j);
        }
        for(size_t j = 0; j < m; ++j){
            y->at(j,i) = y->at(j,i-1) + h*(t->at(j,0)+2.0*t->at(j,1)+2.0*t->at(j,2)+t->at(j,3))/6.0;
        }
    }
}
