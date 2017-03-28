/* 
 * Author: Johannes M Dieterich
 */

#ifndef CARTESIANGRID_HPP
#define	CARTESIANGRID_HPP

#include <armadillo>
#include <cmath>
#include <memory>
#include <complex.h>
#include <tgmath.h>
#include "BasicGridComputer.hpp"
#include "FourierGrid.hpp"
#include "GVectorBuilder.hpp"
using namespace std;
using namespace arma;

class CartesianGrid: public FourierGrid {
public:
    
    virtual ~CartesianGrid(){
#ifdef LIBKEDF_DEBUG
        cout << "DEBUG: destructor for Cartesian grid called." << endl;
#endif
        _gNorms.reset();
    }
    
    void multiplyGNorms(){
        
#ifdef LIBKEDF_DEBUG
        cout << "DEBUG: multiplying gNorms in Cartesian grid." << endl;
#endif
    
        // do a Fourier operation
        cx_cube* rec = this->getReciprocalGrid();
        
        const cube* gNorms = this->getGNorms();
        
        const uword elems = rec->n_elem;
        
        #pragma omp parallel for default(none) shared(rec,gNorms)
        for(uword x = 0; x < elems; ++x){
            const double norm = gNorms->at(x);
            rec->at(x) *= -norm*norm;
        }
        this->completeReciprocal(rec);

#ifdef LIBKEDF_DEBUG
        cout << "DEBUG: Done multiplying gNorms in Cartesian grid." << endl;
#endif
    }
    
    void multiplyGVectorsX(){
        
        cx_cube* rec = getReciprocalGrid();
        const cube* gVecs = getGVectorsX();
    
        const uword elems = rec->n_elem;
        
        const complex<double> imag(0,1);
        
        #pragma omp parallel for default(none) shared(rec,gVecs)
        for(uword x = 0; x < elems; ++x){
            rec->at(x) *= imag*gVecs->at(x);;
        }
        completeReciprocal(rec);
    }
    
    void multiplyGVectorsY(){
        
        cx_cube* rec = getReciprocalGrid();
        const cube* gVecs = getGVectorsY();
    
        const uword elems = rec->n_elem;
        
        const complex<double> imag(0,1);
        
        #pragma omp parallel for default(none) shared(rec,gVecs)
        for(uword x = 0; x < elems; ++x){
            rec->at(x) *= imag*gVecs->at(x);
        }
        completeReciprocal(rec);
    }
    
    void multiplyGVectorsZ(){
        
        cx_cube* rec = getReciprocalGrid();
        const cube* gVecs = getGVectorsZ();
    
        const uword elems = rec->n_elem;
        
        const complex<double> imag(0,1);
        
        #pragma omp parallel for default(none) shared(rec,gVecs)
        for(uword x = 0; x < elems; ++x){
            rec->at(x) *= imag*gVecs->at(x);
        }
        completeReciprocal(rec);
    }
    
    double integrate() {

        const double sum = sumOver();

        return sum*_cellVolume/_noGridPoints;
    }
    
    double sumOver() {

        const cube* grid = this->readRealGrid();
    
#ifdef _OPENMP
        const size_t nSlices = grid->n_slices;
        const size_t nRows = grid->n_rows;
        const size_t nCols = grid->n_cols;
        double sum = 0.0;
        #pragma omp parallel for default(none) shared(grid,sum)
        for(size_t x = 0; x < nSlices; ++x){
            double tmpSum = 0.0;
            for(size_t col = 0; col < nCols; ++col){
                for(size_t row = 0; row < nRows; ++row){
                    tmpSum += grid->at(row,col,x);
                }
            }
            #pragma omp atomic
            sum += tmpSum;
        }
#else
        const double sum = accu(*grid);
#endif
        
        return sum;
    }
    
    void minMax(double& min, double& max){
        
        const cube* grid = this->readRealGrid();
    
        const size_t nSlices = grid->n_slices;
        const size_t nRows = grid->n_rows;
        const size_t nCols = grid->n_cols;
        
#ifdef _OPENMP
        // parallelize this
        
        double minVals[nSlices];
        double maxVals[nSlices];
        
        #pragma omp parallel for default(none) shared(grid,minVals,maxVals)
        for(size_t x = 0; x < nSlices; ++x){
            minVals[x] = 1e42;
            maxVals[x] = 0.0;
            for(size_t col = 0; col < nCols; ++col){
                for(size_t row = 0; row < nRows; ++row){
                    const double d = grid->at(row,col,x);
                    if(d < minVals[x]){
                        minVals[x] = d;
                    } else if(d > maxVals[x]){
                        maxVals[x] = d;
                    }
                }
            }
        }
        
        // do the aggregation
        double myMin = 1e42;
        double myMax = 0.0;
        for(size_t x = 0; x < nSlices; ++x){
            if(minVals[x] < myMin){
                myMin = minVals[x];
            }
            if(maxVals[x] > myMax){
                myMax = maxVals[x];
            }
        }
#else
        double myMin = 1e42;
        double myMax = 0.0;
        for(size_t x = 0; x < nSlices; ++x){
            for(size_t col = 0; col < nCols; ++col){
                for(size_t row = 0; row < nRows; ++row){
                    const double d = grid->at(row,col,x);
                    myMin = std::min(myMin,d);
                    myMax = std::max(myMax,d);
                }
            }
        }
#endif
        
        min = myMin;
        max = myMax;
    }
    
    const cube* getGNorms() {
        
        // lazy init
        if(_gNorms == NULL){
            setupGVectors(_cellVectors);
        }
        
        return _gNorms.get();
    }
    
    const cube* getGVectorsX(){
        
        // lazy init
        if(_gVectorsX == NULL){
            setupGVectors(_cellVectors);
        }
        
        return _gVectorsX.get();
    }
    
    const cube* getGVectorsY(){
        
        // lazy init
        if(_gVectorsY == NULL){
            setupGVectors(_cellVectors);
        }
        
        return _gVectorsY.get();
    }
    
    const cube* getGVectorsZ(){
        
        // lazy init
        if(_gVectorsZ == NULL){
            setupGVectors(_cellVectors);
        }
        
        return _gVectorsZ.get();
    }
    
    double stressNorm() const {
        return 3*_cellVolume;
    }
    
    size_t getGridPointsX() const {
        return _xDim;
    }
    
    size_t getGridPointsY() const {
        return _yDim;
    }
    
    size_t getGridPointsZ() const {
        return _zDim;
    }
    
    size_t getReciGridPointsX() const {
        return _xRecDim;
    }

    size_t getReciGridPointsY() const {
        return _yRecDim;
    }

    size_t getReciGridPointsZ() const {
        return _zRecDim;
    }

    unsigned long long getTotalGridPoints() const {
        
        const unsigned long long gridX = getGridPointsX();
        const unsigned long long gridY = getGridPointsY();
        const unsigned long long gridZ = getGridPointsZ();
        
        return gridX*gridY*gridZ;
    }
    
    double getCellVolume(){
        return _cellVolume;
    }
    
    double getCellX(){
        return _cellX;
    }
    
    double getCellY(){
        return _cellY;
    }
    
    double getCellZ(){
        return _cellZ;
    }
    
    void updateCellVectors(const double *vecX, const double *vecY, const double *vecZ){
        
        _cellVectors->at(0,0) = vecX[0];
        _cellVectors->at(0,1) = vecX[1];
        _cellVectors->at(0,2) = vecX[2];
        
        _cellVectors->at(1,0) = vecY[0];
        _cellVectors->at(1,1) = vecY[1];
        _cellVectors->at(1,2) = vecY[2];
        
        _cellVectors->at(2,0) = vecZ[0];
        _cellVectors->at(2,1) = vecZ[1];
        _cellVectors->at(2,2) = vecZ[2];
        
        // following http://mathworld.wolfram.com/Parallelepiped.html
        vec::fixed<3> vecA;
        vec::fixed<3> vecB;
        vec::fixed<3> vecC;
        
        vecA.at(0) = _cellVectors->at(0,0);
        vecA.at(1) = _cellVectors->at(0,1);
        vecA.at(2) = _cellVectors->at(0,2);
        
        vecB.at(0) = _cellVectors->at(1,0);
        vecB.at(1) = _cellVectors->at(1,1);
        vecB.at(2) = _cellVectors->at(1,2);
        
        vecC.at(0) = _cellVectors->at(2,0);
        vecC.at(1) = _cellVectors->at(2,1);
        vecC.at(2) = _cellVectors->at(2,2);
        
        const vec vecBC = cross(vecB, vecC);
        
        _cellVolume = abs(dot(vecA, vecBC));
        
        _cellX = sqrt(vecA.at(0)*vecA.at(0) + vecA.at(1)*vecA.at(1) + vecA.at(2)*vecA.at(2));
        _cellY = sqrt(vecB.at(0)*vecB.at(0) + vecB.at(1)*vecB.at(1) + vecB.at(2)*vecB.at(2));
        _cellZ = sqrt(vecC.at(0)*vecC.at(0) + vecC.at(1)*vecC.at(1) + vecC.at(2)*vecC.at(2));
        
        this->_gNorms.reset();
        this->_gVectorsX.reset();
        this->_gVectorsY.reset();
        this->_gVectorsZ.reset();
        
        this->_gNorms = NULL;
        this->_gVectorsX = NULL;
        this->_gVectorsY = NULL;
        this->_gVectorsZ = NULL;
    }
    
protected:
    
    CartesianGrid(const size_t xDim, const size_t yDim, const size_t zDim, const shared_ptr<mat> cellVectors)
        : _xDim(xDim), _yDim(yDim), _zDim(zDim), _xRecDim(floor(_xDim/2)+1), // this is due to the arma data format (which we use as basis: col/row/slice in contiguous -> least contiguous order)
        _yRecDim(yDim), _zRecDim(zDim), _noGridPoints(xDim*yDim*zDim), _norm((double) _xDim*_yDim*_zDim),
        _invNorm(1.0/_norm){
                
        _cellVectors = cellVectors;
        
        // following http://mathworld.wolfram.com/Parallelepiped.html
        vec::fixed<3> vecA;
        vec::fixed<3> vecB;
        vec::fixed<3> vecC;
        
        vecA.at(0) = _cellVectors->at(0,0);
        vecA.at(1) = _cellVectors->at(0,1);
        vecA.at(2) = _cellVectors->at(0,2);
        
        vecB.at(0) = _cellVectors->at(1,0);
        vecB.at(1) = _cellVectors->at(1,1);
        vecB.at(2) = _cellVectors->at(1,2);
        
        vecC.at(0) = _cellVectors->at(2,0);
        vecC.at(1) = _cellVectors->at(2,1);
        vecC.at(2) = _cellVectors->at(2,2);
        
        const vec vecBC = cross(vecB, vecC);
        
        _cellVolume = abs(dot(vecA, vecBC));
        
        _cellX = sqrt(vecA.at(0)*vecA.at(0) + vecA.at(1)*vecA.at(1) + vecA.at(2)*vecA.at(2));
        _cellY = sqrt(vecB.at(0)*vecB.at(0) + vecB.at(1)*vecB.at(1) + vecB.at(2)*vecB.at(2));
        _cellZ = sqrt(vecC.at(0)*vecC.at(0) + vecC.at(1)*vecC.at(1) + vecC.at(2)*vecC.at(2));
        
        // initialize to NULL for lazy init
        _gNorms = NULL;
        _gVectorsX = NULL;
        _gVectorsY = NULL;
        _gVectorsZ = NULL;
    }
    
    CartesianGrid(const CartesianGrid& orig) : _xDim(orig._xDim), _yDim(orig._yDim), _zDim(orig._zDim),
        _xRecDim(orig._xRecDim), _yRecDim(orig._yRecDim), _zRecDim(orig._zRecDim),
        _noGridPoints(orig._noGridPoints), _norm(orig._norm), _invNorm(orig._invNorm),
        _cellVolume(orig._cellVolume), _cellX(orig._cellX), _cellY(orig._cellY), _cellZ(orig._cellZ){
    
        _cellVectors = orig._cellVectors;
        
        _gNorms = orig._gNorms;
        _gVectorsX = orig._gVectorsX;
        _gVectorsY = orig._gVectorsY;
        _gVectorsZ = orig._gVectorsZ;
    }
    
    void setupGVectors(const shared_ptr<mat> cellVectors){
        setupSimpleGVectors(cellVectors);
    }
    
    void setupSimpleGVectors(const shared_ptr<mat> cellVectors){
        
        _gNorms = make_shared<cube>(_xRecDim,_yRecDim,_zRecDim);
        
        _gVectorsX = make_shared<cube>(_xRecDim,_yRecDim,_zRecDim);
        _gVectorsY = make_shared<cube>(_xRecDim,_yRecDim,_zRecDim);
        _gVectorsZ = make_shared<cube>(_xRecDim,_yRecDim,_zRecDim);
        
        // construct g norms and vectors
        GVectorBuilder::buildGVectors(cellVectors,_xRecDim,_yRecDim,_zRecDim,_gNorms,_gVectorsX,_gVectorsY,_gVectorsZ);
    }
    
    shared_ptr<mat> _cellVectors;
    shared_ptr<cube> _gNorms;
    shared_ptr<cube> _gVectorsX;
    shared_ptr<cube> _gVectorsY;
    shared_ptr<cube> _gVectorsZ;
    
    const size_t _xDim;
    const size_t _yDim;
    const size_t _zDim;
    const size_t _xRecDim;
    const size_t _yRecDim;
    const size_t _zRecDim;
    const size_t _noGridPoints;
    const double _norm;
    const double _invNorm;
    
    double _cellVolume;
    double _cellX;
    double _cellY;
    double _cellZ;
};


#endif	/* CARTESIANGRID_HPP */

