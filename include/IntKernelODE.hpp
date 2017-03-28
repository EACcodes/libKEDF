/* 
 * Author: Johannes M Dieterich
 */
#ifndef INTKERNELODE_HPP
#define INTKERNELODE_HPP

#include <armadillo>
#include <list>
#include <memory>
#include <rksuite.h>

class IntKernelODE {

public:
    IntKernelODE(bool stats = false);
    ~IntKernelODE();
    std::unique_ptr<arma::mat> makeFirstOrderKernel(ODEKernel* ode, const double wInf);
    std::unique_ptr<arma::mat> makeSecondOrderKernel(ODEKernel* ode, const double wInf);
    const std::shared_ptr<arma::vec> getLastEta() const;
    std::unique_ptr<arma::mat> makeSecondOrderKernel(ODEKernel* ode, const double wInf, const double tstart, const double tend, const double tstep);

private:
    RKSUITE* _rksuite;
    bool _stats;
    std::shared_ptr<arma::vec> _eta;
    
    const double MAGICINIT = -4242.0;
    const double MAXT = 100.0;
    const double TSTART = 1e-3;
    const double TEND = 3.0;
    const double TSTEP = 1e-3;
    const double THRESH = 1e-10;
};

#endif /* INTKERNELODE_HPP */

