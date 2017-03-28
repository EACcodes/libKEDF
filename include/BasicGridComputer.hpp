/* 
 * Author: Johannes M Dieterich
 */

#ifndef BASICGRIDCOMPUTER_HPP
#define BASICGRIDCOMPUTER_HPP

using namespace arma;
using namespace std;

class GridComputer{
    
public:
        
    template<class GridType>
    static void multiplyElementwise(GridType * grid, GridType * otherGrid){
        
#ifdef LIBKEDF_DEBUG
        cout << "DEBUG: multiply called in grid basic computer." << endl;
#endif
        
        cube* real = grid->getRealGrid();
        const cube* other = otherGrid->readRealGrid();
        
        const uword elems = real->n_elem;

#ifndef LIBKEDF_NO_ALIGNED_MEMORY
        #pragma omp parallel for simd default(none) shared(real, other)
#else
        #pragma omp parallel for default(none) shared(real, other)
#endif
        for(uword x = 0; x < elems; ++x){
            real->at(x) *= other->at(x);
        }
        grid->complete(real);
      
#ifdef LIBKEDF_DEBUG
        cout << "DEBUG: multiply finished in grid base class." << endl;
#endif
    }

    template<class GridType>
    static unique_ptr<GridType> laplacian(GridType * grid) {
        
#ifdef LIBKEDF_DEBUG
        cout << "DEBUG: doing laplacian calculation in basic grid computer." << endl;
#endif
        unique_ptr<GridType> laplacian(grid->duplicate());

        laplacian->multiplyGNorms();
        
        return move(laplacian);
    }
    
    template<class GridType>
    static unique_ptr<GridType> gradientSquared(GridType * grid) {
        
#ifdef LIBKEDF_DEBUG
        cout << "DEBUG: doing gradient squared calculation in basic grid computer." << endl;
#endif
        
        unique_ptr<GridType> gradientSquared = grid->emptyDuplicate();
        cube* real = gradientSquared->getRealGrid();
    
        const uword elems = real->n_elem;
        
        // we will need this for each directional term
        unique_ptr<GridType> recGrid = grid->duplicate();
        cx_cube* recCube = recGrid->getReciprocalGrid();
        recGrid->completeReciprocal(recCube);
        
        // X-contribution
        unique_ptr<GridType> workGrid = recGrid->duplicate();
        workGrid->multiplyGVectorsX(); // now contains g-Vectors-X*FFT(density)
        
        const cube* xReal = workGrid->readRealGrid();
        
#ifndef LIBKEDF_NO_ALIGNED_MEMORY
        #pragma omp parallel for simd default(none) shared(real, xReal)
#else
        #pragma omp parallel for default(none) shared(real, xReal)
#endif
        for(uword x = 0; x < elems; ++x){
                    
            const double d = xReal->at(x);
            real->at(x) = d*d;
        }        
        // Y-contribution
        workGrid->copyStateIn(recGrid.get());
        workGrid->multiplyGVectorsY(); // now contains g-Vectors-Y*FFT(density)
        
        const cube* yReal = workGrid->readRealGrid();
        
#ifndef LIBKEDF_NO_ALIGNED_MEMORY
        #pragma omp parallel for simd default(none) shared(real, yReal)
#else
        #pragma omp parallel for default(none) shared(real, yReal)
#endif
        for(uword x = 0; x < elems; ++x){
                    
            const double d = yReal->at(x);
            real->at(x) += d*d;
        }
        
        // Z-contribution
        workGrid->copyStateIn(recGrid.get());
        workGrid->multiplyGVectorsZ(); // now contains g-Vectors-Z*FFT(density)
        
        const cube* zReal = workGrid->readRealGrid();
        
#ifndef LIBKEDF_NO_ALIGNED_MEMORY
        #pragma omp parallel for simd default(none) shared(real, zReal)
#else
        #pragma omp parallel for default(none) shared(real, zReal)
#endif
        for(uword x = 0; x < elems; ++x){
                    
            const double d = zReal->at(x);
            real->at(x) += d*d;
        }
        workGrid.reset();
        recGrid.reset();
        
        // finished
        gradientSquared->complete(real);
        
        return move(gradientSquared);
    }
    
private:
    GridComputer(){}
};

#endif /* BASICGRIDCOMPUTER_HPP */

