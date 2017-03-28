/* 
 * Author: Johannes M Dieterich
 */

#ifndef GRIDFACTORY_HPP
#define	GRIDFACTORY_HPP

#include "FourierGrid.hpp"
using namespace std;

class GridFactory{
public:
    static FourierGrid* constructFourierGrid(const size_t globX, const size_t globY,
            const size_t globZ, const shared_ptr<mat> cellVectors, const string config);
private:
};

#endif	/* GRIDFACTORY_HPP */

