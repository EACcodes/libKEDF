/* 
 * Author: Johannes M Dieterich
 */

#ifndef WANGGOVINDCARTER_HPP
#define WANGGOVINDCARTER_HPP

#include <armadillo>
#include <array>
#include <memory>
#include "KEDF.hpp"
#include "IntKernelODE.hpp"
using namespace std;
using namespace arma;

#include "TayloredWangGovindCarter.hpp"

class WangGovindCarterODE : public ODEKernel {
    
public:
    WangGovindCarterODE(const double beta, const double gamma);
    ~WangGovindCarterODE();
    void evaluate(const double t, double y[], double yp[]) override;
    
private:
    double _beta;
    double _gamma;
};

class AnalyticalWangGovindCarterKernel {
    
public:
    AnalyticalWangGovindCarterKernel(const double alpha, const double beta, const double gamma, const size_t numTermsAB = 100);
    ~AnalyticalWangGovindCarterKernel();
    
    void fillWGCKernel(unique_ptr<cx_cube>& kernel, unique_ptr<cx_cube>& betaKernel, const cube* gNorms);
    
private:
    
    void elementWGC(const double eta, array<int,3>& w);
    
    void fillAB();
    
    size_t _numTermsAB;
    double _pea;
    double _cue;
    double _ell;
    double _cOne;
    double _cTwo;
    double _aM1;
    vector<double> _an;
    vector<double> _bn;
};

class NumericalWangGovindCarterKernel {
    
public:
    NumericalWangGovindCarterKernel(const double alpha, const double beta, const double gamma, const double rhoS);
    ~NumericalWangGovindCarterKernel();
    
    void fillWGCKernel(cube* kernel0th, const cube* gNorms);
    
    void fillWGCKernel(cube* kernel0th, cube* kernel1st, const cube* gNorms);
    
    void fillWGCKernel(cube* kernel0th, cube* kernel1st, cube* kernel2nd, const cube* gNorms);
    
    void fillWGCKernel(cube* kernel0th, cube* kernel1st, cube* kernel2nd, cube* kernel3rd, const cube* gNorms);
    
private:
    double _tkFStar;
    double _alpha;
    double _beta;
    double _gamma;
    double _rhoS;
    
    unique_ptr<mat> _w;
    shared_ptr<vec> _eta;
    int _nVals;
    double *_nls_wpp;
    double *_nls_w1pp;
    double *_nls_w2pp;
};

class NumericalRealSpaceWangGovindCarterKernel {
    
public:
    NumericalRealSpaceWangGovindCarterKernel();
    ~NumericalRealSpaceWangGovindCarterKernel();
    
private:
    double _tkFStar;
    double _alpha;
    double _beta;
    double _gamma;
    double _rhoS;
    
    unique_ptr<mat> _w;
    shared_ptr<vec> _eta;
    int _nVals;
    shared_ptr<vec> _nls_wpp;
    shared_ptr<vec> _nls_w1pp;
    shared_ptr<vec> _nls_w2pp;
};

#endif /* WANGGOVINDCARTER_HPP */

