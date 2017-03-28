/* 
 * Author: Johannes M Dieterich
 */

#ifndef FOURIERGRID_HPP
#define	FOURIERGRID_HPP

#include <complex>
#include <armadillo>
#include <memory>
#include "Grid.hpp"
using namespace std;
using namespace arma;

class FourierGrid: public Grid {
    
public:
    
    virtual ~FourierGrid(){};
    
    virtual unique_ptr<FourierGrid> createFourierDuplicate() const = 0;
    
    virtual unique_ptr<FourierGrid> createFourierEmptyDuplicate() const = 0;
    
    virtual cx_cube* getReciprocalGrid() = 0;
    
    virtual const cx_cube* readReciprocalGrid() = 0;
    
    virtual const cube* getGNorms() = 0;
    
    virtual const cube* getGVectorsX() = 0;
    
    virtual const cube* getGVectorsY() = 0;
    
    virtual const cube* getGVectorsZ() = 0;
    
    virtual void completeReciprocal(cx_cube* reciprocalGrid) = 0;
    
    virtual void resetToReal() = 0;
    
    virtual void resetToReciprocal() = 0;
    
    virtual void multiplyGNorms() = 0;
            
private:
};

#endif	/* FOURIERGRID_HPP */

