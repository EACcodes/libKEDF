/* 
 * Author: Johannes M Dieterich
 */

#ifndef STRESSTENSOR_HPP
#define	STRESSTENSOR_HPP

#include <armadillo>
using namespace std;
using namespace arma;

class StressTensor {
    
public:
    StressTensor();
    ~StressTensor();
    
    mat* getTensor();
private:
    mat* _tensor;
};

#endif	/* STRESSTENSOR_HPP */

