/* 
 * Author: Johannes M Dieterich
 */

#ifndef KEDF_HPP
#define	KEDF_HPP

#include <armadillo>
#include <memory>
#include <string>
#include "Grid.hpp"
#include "StressTensor.hpp"
using namespace std;

template<class GridType>
class KEDF {
public:
    virtual ~KEDF(){};
    virtual string getMethodDescription() const = 0;
    virtual vector<string> getCitations() const = 0;
    virtual vector<string> getWorkingEquations() const = 0;
    virtual double calcEnergy(const GridType& grid) const = 0;
    virtual double calcPotential(const GridType& grid, GridType& potential) const = 0;
    virtual unique_ptr<StressTensor> calcStress(const GridType& grid) const = 0;
private:
};

#endif	/* KEDF_HPP */

