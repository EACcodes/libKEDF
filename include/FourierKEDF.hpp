/* 
 * Author: Johannes M Dieterich
 */

#ifndef FOURIERKEDF_HPP
#define	FOURIERKEDF_HPP

#include <armadillo>
#include <memory>
#include <string>
#include "FourierGrid.hpp"
#include "Grid.hpp"
#include "StressTensor.hpp"
using namespace std;

class FourierKEDF {
public:
    virtual ~FourierKEDF(){};
    virtual string getMethodDescription() const = 0;
    virtual vector<string> getCitations() const = 0;
    virtual vector<string> getWorkingEquations() const = 0;
    virtual double calcEnergy(const FourierGrid& grid) const = 0;
    virtual double calcPotential(const FourierGrid& grid, FourierGrid& potential) const = 0;
    virtual unique_ptr<StressTensor> calcStress(const FourierGrid& grid) const = 0;
private:
};

#endif	/* FOURIERKEDF_HPP */

