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
#include "IntKernelODE.hpp"
#include <cmath>
#include <memory>
#include <stdexcept>
#include <iostream>
using namespace arma;
using namespace std;

IntKernelODE::IntKernelODE(bool stats){
    _rksuite = new RKSUITE();
    _stats = stats;
}

IntKernelODE::~IntKernelODE(){;
    delete _rksuite;
}

unique_ptr<mat> IntKernelODE::makeFirstOrderKernel(ODEKernel* ode, const double wInf){
    
    // some defaults
    const int neq = 1;
    const int method = 2;
    const bool errass = false;
    const bool message = false;
    const double hstart = 0.0;
    const double tol = 5e-5;
    char* task = new char[1];
    task[0] = 'U';
    
    // construct eta
    double* t_start = new double[neq];
    double* t_end = new double[neq];
    double* t_step = new double[neq];
    
    t_start[0] = 0.0;
    t_end[0] = 50.0; 
    t_step[0] = 0.001;
    
    const size_t t_num = floor(t_end[0] / t_step[0]) + 1;
    _eta = make_shared<vec>(t_num);
    
    for(size_t i = 0; i < t_num; ++i){
        _eta->at(i) = t_start[0] + i*t_step[0];
    }
    
    unique_ptr<mat> w = make_unique<mat>(t_num,2);
    w->fill(MAGICINIT);

    double tend = _eta->at(1)*0.9;
    double tstart = _eta->at(t_num-1);
    
    double* ystart = new double[neq];
    ystart[0] = wInf;
    
    double* thresh = new double[neq];
    thresh[0] = THRESH;
    
    // setup RKSUITE
    _rksuite->setup(neq, tstart, ystart, tend, tol, thresh, method, task, 
            errass, hstart, message);
    
    // define the last points
    w->at(t_num-1,0) = ystart[0];
    w->at(t_num-1,1) = 0.0;
    
    double* y = new double[neq];
    double* yp = new double[neq];
    double* ymax = new double[neq];
    double t = -42.0;
    
    // we loop from top to bottom
    for(int i = t_num-2; i > 0; --i){ // do not go to 0!
        
        const double twant = _eta->at(i);
        
        int uflag = -2;
        do{
            
            _rksuite->ut(ode, twant, t, y, yp, ymax, uflag);
            
            if(uflag == 1){
                // success
                w->at(i,0) = y[0];
                w->at(i,1) = yp[0];
            } else if(uflag == 4){
                throw runtime_error("RKSuite signals problem is too stiff.");
            } else if(uflag == 5){
                throw runtime_error("RKSuite signals problem is too demanding for requested accuracy.");
            } else if(uflag == 6){
                throw runtime_error("RKSuite signals global error assessment may not be valid beyond current point..");
            } else if(uflag == 911){
                throw runtime_error("RKSuite signals an emergency (911).");
            }         
            
        } while(uflag != 1);
    }
    
    w->at(0,0) = w->at(1,0);
    w->at(0,1) = w->at(1,1);
    
    // make sure all the w/w1 are filled
    for(size_t i = 0; i < t_num; ++i){
        
        if(w->at(i,0) == MAGICINIT) {throw runtime_error("Element in w[0] is not initialized.");}
        if(w->at(i,1) == MAGICINIT) {throw runtime_error("Element in w[1] is not initialized.");}
    }
    
    
    if(_stats){
        // some stats
        int noSteps = -1;
        int cost = -1;
        double waste = -42.0;
        int noAccepted = -1;
        double nextStepSize = -1;
        _rksuite->stat(noSteps, cost, waste, noAccepted, nextStepSize);
        
        cout << "ODE statistics from RKSuite:  " << endl;
        cout << "    #function calls:          " << noSteps << endl;
        cout << "    cost per derivative call: " << cost << endl;
        cout << "    #steps wasted:            " << waste << endl;
        cout << "    #steps OK:                " << noAccepted << endl;
        cout << "    next step size:           " << nextStepSize << endl;
    }
    
    delete[] y;
    delete[] yp;
    delete[] ymax;
    delete[] t_start;
    delete[] t_end;
    delete[] t_step;
    delete[] task;
    delete[] ystart;
    delete[] thresh;
    
    // correct w[1] values with the eta values
    w->at(0,0) = 0.0;
    double *w1 = w->colptr(1);
    
    #pragma omp parallel for default(none) shared(w1)
    for(size_t x = 0; x < t_num; ++x){
        const double d = _eta->at(x);
        w1[x] *= d;
    }
    
    return w;
}

unique_ptr<mat> IntKernelODE::makeSecondOrderKernel(ODEKernel* ode, const double wInf){
    
    // some defaults
    const int neq = 2;
    const int method = 2;
    const bool errass = false;
    const bool message = false;
    const double hstart = 0.0;
    const double tol = 5e-5;
    char* task = new char[1];
    task[0] = 'U';
    
    // construct eta
    double* t_start = new double[neq];
    double* t_end = new double[neq];
    double* t_step = new double[neq];
    
    t_start[0] = TSTART; 
    t_end[0] = TEND; 
    t_step[0] = TSTEP;

    t_start[1] = t_end[0] + t_step[0];
    t_end[1] = MAXT;
    t_step[1] = t_step[0]*10.0;
    
    size_t pointSum = 0;
    int* npoints = new int[neq];
    for(int i = 0; i < neq; ++i){
        npoints[i] =  floor((t_end[i]-t_start[i])/t_step[i]) - 1;
        pointSum += npoints[i];
    }
    
    unique_ptr<mat> w = make_unique<mat>(pointSum,3);
    w->fill(MAGICINIT);
    
    
    _eta = make_shared<vec>(pointSum);
    for(size_t i = 0; i < pointSum; ++i){
        _eta->at(i) = -1.0;
    }
    
    for(int i = 0; i < npoints[0]; ++i){
        _eta->at(i) = t_start[0] + i*t_step[0];
    }
    
    for(size_t i = npoints[0]; i < pointSum; ++i){
        _eta->at(i) = t_start[1] + (i-npoints[0])*t_step[1];
    }
    
    // data
    double tend = _eta->at(0)*0.9;
    double tstart = _eta->at(pointSum-1);
    
    double* ystart = new double[neq];
    ystart[0] = wInf;
    ystart[1] = 0.0;
    
    double* thresh = new double[neq];
    for(int x = 0; x < neq; ++x){
        thresh[x] = THRESH;
    }
    
    // setup RKSUITE
    _rksuite->setup(neq, tstart, ystart, tend, tol, thresh, method, task, 
            errass, hstart, message);
    
    // define the last points
    w->at(pointSum-1,0) = 0.0;
    w->at(pointSum-1,1) = 0.0;
    w->at(pointSum-1,2) = 0.0;
    
    double* y = new double[neq];
    double* yp = new double[neq];
    double* ymax = new double[neq];
    double t = -42.0;
    
    // we loop from top to bottom
    for(int i = pointSum-2; i >= 0; --i){
        
        const double twant = _eta->at(i);
        
        int uflag = -2;
        do{
            
            _rksuite->ut(ode, twant, t, y, yp, ymax, uflag);
            
            if(uflag == 1){
                // success
                w->at(i,0) = y[0];
                w->at(i,1) = y[1];
                w->at(i,2) = yp[1];
            } else if(uflag == 4){
                throw runtime_error("RKSuite signals problem is too stiff.");
            } else if(uflag == 5){
                throw runtime_error("RKSuite signals problem is too demanding for requested accuracy.");
            } else if(uflag == 6){
                throw runtime_error("RKSuite signals global error assessment may not be valid beyond current point..");
            } else if(uflag == 911){
                throw runtime_error("RKSuite signals an emergency (911).");
            }         
            
        } while(uflag != 1);
    }
    
    // make sure all the w/w1/w2 are filled
    for(size_t i = 0; i < pointSum; ++i){
        
        if(w->at(i,0) == MAGICINIT) {throw runtime_error("Element in w[0] is not initialized.");}
        if(w->at(i,1) == MAGICINIT) {throw runtime_error("Element in w[1] is not initialized.");}
        if(w->at(i,2) == MAGICINIT) {throw runtime_error("Element in w[2] is not initialized.");}
    }
    
    
    if(_stats){
        // some stats
        int noSteps = -1;
        int cost = -1;
        double waste = -42.0;
        int noAccepted = -1;
        double nextStepSize = -1;
        _rksuite->stat(noSteps, cost, waste, noAccepted, nextStepSize);
        
        cout << "ODE statistics from RKSuite:  " << endl;
        cout << "    #function calls:          " << noSteps << endl;
        cout << "    cost per derivative call: " << cost << endl;
        cout << "    #steps wasted:            " << waste << endl;
        cout << "    #steps OK:                " << noAccepted << endl;
        cout << "    next step size:           " << nextStepSize << endl;
    }
    
    delete[] y;
    delete[] yp;
    delete[] ymax;
    delete[] t_start;
    delete[] t_end;
    delete[] t_step;
    delete[] task;
    delete[] npoints;
    delete[] ystart;
    delete[] thresh;
    
    // correct w[1] and w[2] values with the eta values
    double *w1 = w->colptr(1);
    double *w2 = w->colptr(2);
    
    #pragma omp parallel for default(none) shared(pointSum,w1,w2)
    for(size_t x = 0; x < pointSum; ++x){
        const double d = _eta->at(x);
        w1[x] *= d;
        w2[x] *= d*d;
    }
    
    return w;
}

unique_ptr<mat> IntKernelODE::makeSecondOrderKernel(ODEKernel* ode, const double wInf, const double tstart, const double tend, const double tstep){
    
    // some defaults
    const int neq = 2;
    const int method = 2;
    const bool errass = false;
    const bool message = false;
    const double hstart = 0.0;
    const double tol = 5e-5;
    char* task = new char[1];
    task[0] = 'U';
    
    // construct eta
    double* t_start = new double[neq];
    double* t_end = new double[neq];
    double* t_step = new double[neq];
    
    t_start[0] = tstart; 
    t_end[0] = tend; 
    t_step[0] = tstep;

    t_start[1] = t_end[0] + t_step[0];
    t_end[1] = MAXT;
    t_step[1] = t_step[0]*10.0;
    
    size_t pointSum = 0;
    int* npoints = new int[neq];
    for(int i = 0; i < neq; ++i){
        npoints[i] =  floor((t_end[i]-t_start[i])/t_step[i]) - 1;
        pointSum += npoints[i];
    }
    
    unique_ptr<mat> w = make_unique<mat>(pointSum,3);
    w->fill(MAGICINIT);
    
    
    _eta = make_shared<vec>(pointSum);
    for(size_t i = 0; i < pointSum; ++i){
        _eta->at(i) = -1.0;
    }
    
    for(int i = 0; i < npoints[0]; ++i){
        _eta->at(i) = t_start[0] + i*t_step[0];
    }
    
    for(size_t i = npoints[0]; i < pointSum; ++i){
        _eta->at(i) = t_start[1] + (i-npoints[0])*t_step[1];
    }
    
    // data
    double tendX = _eta->at(0)*0.9;
    double tstartX = _eta->at(pointSum-1);
    
    double* ystart = new double[neq];
    ystart[0] = wInf;
    ystart[1] = 0.0;
    
    double* thresh = new double[neq];
    for(int x = 0; x < neq; ++x){
        thresh[x] = THRESH;
    }
    
    // setup RKSUITE
    _rksuite->setup(neq, tstartX, ystart, tendX, tol, thresh, method, task, 
            errass, hstart, message);
    
    // define the last points
    w->at(pointSum-1,0) = 0.0;
    w->at(pointSum-1,1) = 0.0;
    w->at(pointSum-1,2) = 0.0;
    
    double* y = new double[neq];
    double* yp = new double[neq];
    double* ymax = new double[neq];
    double t = -42.0;
    
    // we loop from top to bottom
    for(int i = pointSum-2; i >= 0; --i){
        
        const double twant = _eta->at(i);
        
        int uflag = -2;
        do{
            
            _rksuite->ut(ode, twant, t, y, yp, ymax, uflag);
            
            if(uflag == 1){
                // success
                w->at(i,0) = y[0];
                w->at(i,1) = y[1];
                w->at(i,2) = yp[1];
            } else if(uflag == 4){
                throw runtime_error("RKSuite signals problem is too stiff.");
            } else if(uflag == 5){
                throw runtime_error("RKSuite signals problem is too demanding for requested accuracy.");
            } else if(uflag == 6){
                throw runtime_error("RKSuite signals global error assessment may not be valid beyond current point..");
            } else if(uflag == 911){
                throw runtime_error("RKSuite signals an emergency (911).");
            }         
            
        } while(uflag != 1);
    }
    
    // make sure all the w/w1/w2 are filled
    for(size_t i = 0; i < pointSum; ++i){
        
        if(w->at(i,0) == MAGICINIT) {throw runtime_error("Element in w[0] is not initialized.");}
        if(w->at(i,1) == MAGICINIT) {throw runtime_error("Element in w[1] is not initialized.");}
        if(w->at(i,2) == MAGICINIT) {throw runtime_error("Element in w[2] is not initialized.");}
    }
    
    
    if(_stats){
        // some stats
        int noSteps = -1;
        int cost = -1;
        double waste = -42.0;
        int noAccepted = -1;
        double nextStepSize = -1;
        _rksuite->stat(noSteps, cost, waste, noAccepted, nextStepSize);
        
        cout << "ODE statistics from RKSuite:  " << endl;
        cout << "    #function calls:          " << noSteps << endl;
        cout << "    cost per derivative call: " << cost << endl;
        cout << "    #steps wasted:            " << waste << endl;
        cout << "    #steps OK:                " << noAccepted << endl;
        cout << "    next step size:           " << nextStepSize << endl;
    }
    
    delete[] y;
    delete[] yp;
    delete[] ymax;
    delete[] t_start;
    delete[] t_end;
    delete[] t_step;
    delete[] task;
    delete[] npoints;
    delete[] ystart;
    delete[] thresh;
    
    // correct w[1] and w[2] values with the eta values
    double *w1 = w->colptr(1);
    double *w2 = w->colptr(2);
    
    #pragma omp parallel for default(none) shared(pointSum,w1,w2)
    for(size_t x = 0; x < pointSum; ++x){
        const double d = _eta->at(x);
        w1[x] *= d;
        w2[x] *= d*d;
    }
    
    return w;
}

const shared_ptr<arma::vec> IntKernelODE::getLastEta() const {
    return _eta;
}
