/* 
 * Author: Johannes M Dieterich
 */

#ifndef HELPERFUNCTIONS_HPP
#define	HELPERFUNCTIONS_HPP

#include <armadillo>
#include <memory>
#include "Grid.hpp"
using namespace std;
using namespace arma;

class CutoffFunctions {

public:
    template<class GridComputer, template<class> class GridType>
    static void WGCVacuumCutoff(unique_ptr<GridType<GridComputer> >& density, const double rhoV, const double rhoStep);
    
    static inline double vacuumCutoff(const double density, const double rhoV, const double rhoStep);
    
    template<class GridComputer, template<class> class GridType>
    static void WGCVacuumCutoffDeriv(unique_ptr<GridType<GridComputer> >& density, const double rhoV, const double rhoStep);
    
    static inline double vacuumCutoffDeriv(const double density, const double rhoV, const double rhoStep);
};


class MathFunctions {

public:
    static double lindhardResponse(const double eta, const double lambda, const double mu);
    
    static double derivativeLindhardResponse(const double eta, const double mu);
};

class MemoryFunctions {
public:
    static unique_ptr<cube, void (*) (cube*)> allocateScratch(const size_t rows, const size_t cols, const size_t slices);
    static unique_ptr<fcube, void (*) (fcube*)> allocateScratchFloat(const size_t rows, const size_t cols, const size_t slices);
    static unique_ptr<cx_cube, void (*) (cx_cube*)> allocateReciprocalScratch(const size_t rows, const size_t cols, const size_t slices);
    static unique_ptr<vec, void (*) (vec*)> allocateScratch(const size_t elements);
    static cube* allocateScratchCube(const size_t rows, const size_t cols, const size_t slices);
    static fcube* allocateScratchCubeFloat(const size_t rows, const size_t cols, const size_t slices);
    static cx_cube* allocateReciprocalScratchCube(const size_t rows, const size_t cols, const size_t slices);
};

#endif	/* HELPERFUNCTIONS_HPP */

