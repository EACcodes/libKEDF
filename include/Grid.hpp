/* 
 * Author: Johannes M Dieterich
 */

#ifndef GRID_HPP
#define	GRID_HPP

#include <cmath>
#include <complex>
#include <armadillo>
#include <memory>
#include <stdio.h>
using namespace std;
using namespace arma;

class Grid {
    
public:
    
    virtual ~Grid(){};
    
    virtual cube* getRealGrid() = 0;
    
    virtual const cube* tryReadRealGrid() const = 0;
    
    virtual const cube* readRealGrid() = 0;
    
    virtual void complete(cube* realGrid) = 0;
    
    virtual double sumOver() = 0;
    
    virtual double integrate() = 0;
    
    virtual void minMax(double& min, double& max) = 0;
    
    virtual double stressNorm() const = 0;
    
    virtual size_t getGridPointsX() const = 0;
    
    virtual size_t getGridPointsY() const = 0;
    
    virtual size_t getGridPointsZ() const = 0;
    
    virtual unsigned long long getTotalGridPoints() const = 0;
    
    void convolve(Grid * otherGrid){}
    
    virtual void sqrtGrid() {
        
#ifdef LIBKEDF_DEBUG
        cout << "DEBUG: sqrtGrid called in grid base class." << endl;
#endif
        
        cube* data = this->getRealGrid();
        const size_t nSlices = data->n_slices;
        const size_t nRows = data->n_rows;
        const size_t nCols = data->n_cols;
        #pragma omp parallel for default(none) shared(data)
        for(size_t x = 0; x < nSlices; ++x){
            for(size_t col = 0; col < nCols; ++col){
                for(size_t row = 0; row < nRows; ++row){
                    data->at(row,col,x) = sqrt(data->at(row,col,x));
                }
            }
        }
        this->complete(data);
        
#ifdef LIBKEDF_DEBUG
        cout << "DEBUG: sqrtGrid finished in grid base class." << endl;
#endif
        
    }
    
    virtual void powGrid(const double exponent){
        
#ifdef LIBKEDF_DEBUG
        cout << "DEBUG: powGrid called in grid base class." << endl;
#endif
        
        cube* data = this->getRealGrid();
        const size_t nSlices = data->n_slices;
        const size_t nRows = data->n_rows;
        const size_t nCols = data->n_cols;
        #pragma omp parallel for default(none) shared(data)
        for(size_t x = 0; x < nSlices; ++x){
            for(size_t col = 0; col < nCols; ++col){
                for(size_t row = 0; row < nRows; ++row){
                    data->at(row,col,x) = pow(data->at(row,col,x), exponent);
                }
            }
        }
        this->complete(data);
        
#ifdef LIBKEDF_DEBUG
        cout << "DEBUG: powGrid finished in grid base class." << endl;
#endif
    }
    
    virtual void updateRealGrid(const double* rawData){
    
        cube* grid = this->getRealGrid();
        
        double* gridData = grid->memptr();
        
        const size_t nSlices = this->getGridPointsZ();
        const size_t nRows = this->getGridPointsY();
        const size_t nCols = this->getGridPointsX();
        const size_t totSize = nSlices*nRows*nCols*sizeof(double);
        
        memcpy(gridData, rawData, totSize);
    
        this->complete(grid);
    }
    
    virtual void getRealGridData(double* rawData){
    
        const cube* grid = this->readRealGrid();
        
        const double* gridData = grid->memptr();
        
        const size_t nSlices = this->getGridPointsZ();
        const size_t nRows = this->getGridPointsY();
        const size_t nCols = this->getGridPointsX();
        const size_t totSize = nSlices*nRows*nCols*sizeof(double);
        
        memcpy(rawData, gridData, totSize);
    }
    
    virtual void finalize(){
    }
    
    constexpr static double NUMERICALACCURACY = 1e-10;
        
private:
};

#endif	/* GRID_HPP */

