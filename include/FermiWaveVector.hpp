/* 
 * Author: Johannes M Dieterich
 */

#ifndef FERMIWAVEVECTOR_HPP
#define FERMIWAVEVECTOR_HPP

class FermiWaveVector {

public:
    
    inline static double nonLocalTwoBody(const double rho1, const double rho2, const double invGamma);
    inline static double localOneBody(const double rho);
};

#endif /* FERMIWAVEVECTOR_HPP */

