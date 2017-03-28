/* 
 * Author: Johannes M Dieterich
 */

#ifndef CARTESIANOOPGRID_HPP
#define	CARTESIANOOPGRID_HPP

#include <complex>
#include <fftw3.h>
#include <memory>
#include "CartesianGrid.hpp"
using namespace std;

class CartesianOOPGrid: public CartesianGrid {
public:
    
    CartesianOOPGrid(const size_t xDim, const size_t yDim, const size_t zDim, const shared_ptr<mat> cellVectors);
    ~CartesianOOPGrid();
    
    unique_ptr<CartesianOOPGrid> duplicate() const;
    unique_ptr<CartesianOOPGrid> emptyDuplicate() const;
    
    void copyStateIn(const CartesianOOPGrid * fromGrid);
    
    void addGrid(CartesianOOPGrid *other);
    void fmaGrid(const double mult, CartesianOOPGrid *other);
    
    unique_ptr<FourierGrid> createFourierDuplicate() const override;
    unique_ptr<FourierGrid> createFourierEmptyDuplicate() const override;
    
    arma::cube* getRealGrid() override;
    arma::cx_cube* getReciprocalGrid() override;
    
    void complete(arma::cube* realGrid) override;
    void completeReciprocal(arma::cx_cube* reciprocalGrid) override;
    
    void resetToReal() override;
    void resetToReciprocal() override;
    
    const arma::cube* readRealGrid() override;
    const arma::cube* tryReadRealGrid() const override;
    const arma::cx_cube* readReciprocalGrid() override;
    
    unique_ptr<CartesianOOPGrid> laplacian() {
        return GridComputer::laplacian<CartesianOOPGrid>(this);
    }
    
    unique_ptr<CartesianOOPGrid> gradientSquared() {
        return GridComputer::gradientSquared<CartesianOOPGrid>(this);
    }
    
    unique_ptr<CartesianOOPGrid> directionalDivergenceX(){

#ifdef LIBKEDF_DEBUG
        cout << "DEBUG: doing directional divergence calculation X in Cartesian grid." << endl;
#endif
        
        unique_ptr<CartesianOOPGrid> divergenceG(this->duplicate());
        divergenceG->multiplyGVectorsX();
        
        return divergenceG;
    }
    
    unique_ptr<CartesianOOPGrid> directionalDivergenceY(){
        
#ifdef LIBKEDF_DEBUG
        cout << "DEBUG: doing directional divergence calculation Y in Cartesian grid." << endl;
#endif
        unique_ptr<CartesianOOPGrid> divergenceG(this->duplicate());
        divergenceG->multiplyGVectorsY();
        
        return divergenceG;
    }
    
    unique_ptr<CartesianOOPGrid> directionalDivergenceZ(){
        
#ifdef LIBKEDF_DEBUG
        cout << "DEBUG: doing directional divergence calculation Z in Cartesian grid." << endl;
#endif
        unique_ptr<CartesianOOPGrid> divergenceG(this->duplicate());
        divergenceG->multiplyGVectorsZ();
        
        return divergenceG;
    }
    
    void multiplyElementwise(CartesianOOPGrid* grid){
        GridComputer::multiplyElementwise<CartesianOOPGrid>(this,grid);
    }
    
    void multiplyTwoSqrtOf(CartesianOOPGrid* other);
    
private:
    
    CartesianOOPGrid(const CartesianOOPGrid& orig);
    
    bool _realInSync;
    bool _reciInSync;
    bool _realReturned;
    bool _reciReturned;
    
    double* _rawMemReal;
    fftw_complex* _rawMemReci;
    
    arma::cube* _realGrid;
    arma::cx_cube* _reciGrid;
    
    fftw_plan _planR2C;
    fftw_plan _planC2R;
};

#endif	/* CARTESIANOOPGRID_HPP */

