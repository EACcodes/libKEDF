/* 
 * Copyright (c) 2015-2016, Princeton University, Johannes M Dieterich, Emily A Carter
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation and/or
 * other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its contributors may
 * be used to endorse or promote products derived from this software without specific
 * prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
 * INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
 * NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */
#include <memory>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "CartesianOOPGrid.hpp"
using namespace arma;

CartesianOOPGrid::CartesianOOPGrid(const size_t xDim, const size_t yDim, const size_t zDim, const shared_ptr<mat> cellVectors)
: CartesianGrid(xDim,yDim,zDim,cellVectors){
#ifdef LIBKEDF_DEBUG
    cout << "DEBUG: constructor (I) for OOP grid called." << endl;
#endif
    
    _realInSync = true;
    _reciInSync = true;
    _realReturned = true;
    _reciReturned = true;
    
    // allocate SIMD-enabled scratch space
    const size_t totalDimsReci = _xRecDim*_yRecDim*_zRecDim;
    const size_t totalDimsReal = _xDim*_yDim*_zDim;
    
#ifdef LIBKEDF_NO_ALIGNED_MEMORY
    _rawMemReal = (double*) fftw_malloc(totalDimsReal*sizeof(double));
    _rawMemReci = (fftw_complex*) fftw_malloc(totalDimsReci*sizeof(fftw_complex));
#else
    const size_t alignment = LIBKEDF_ALIGNMENT;
    const int error1 = posix_memalign((void**)&_rawMemReal,alignment,totalDimsReal*sizeof(double));
    const int error2 = posix_memalign((void**)&_rawMemReci,alignment,totalDimsReci*sizeof(complex<double>));
    if(error1 != 0 || error2 != 0){
        cerr << "ERROR: Can't get aligned host memory: " << error1 << " " << error2 << ", size: " << totalDimsReal << " " << totalDimsReci << ", alignment: " << alignment << endl;
        throw runtime_error("No aligned memory could be allocated.");
    }
#endif
    
#ifdef _OPENMP
    // only needed in the *first* setup
    fftw_plan_with_nthreads(omp_get_max_threads());
#endif
    // we use "arma" memory order: i.e. col/row/slice. this is *not* what C-centric codes expect (fastest/contiguous index innermost)
    // FFTW3 manual states that this is n0, n1, n2 where n2 is the halved dimension (i.e., the contiguous one, here: x)
    _planR2C = fftw_plan_dft_r2c_3d(_zDim, _yDim, _xDim, _rawMemReal, _rawMemReci, FFTW_MEASURE | FFTW_DESTROY_INPUT);
    _planC2R = fftw_plan_dft_c2r_3d(_zDim, _yDim, _xDim, _rawMemReci, _rawMemReal, FFTW_MEASURE | FFTW_DESTROY_INPUT);
    
    _realGrid = new cube(_rawMemReal,_xDim,_yDim,_zDim,false,true);
    
    // recast the FFTW type
    complex<double>* tmp = reinterpret_cast<complex<double>*>(_rawMemReci);
    
    _reciGrid = new cx_cube(tmp,_xRecDim,_yRecDim,_zRecDim,false,true);
}

CartesianOOPGrid::CartesianOOPGrid(const CartesianOOPGrid& orig)
: CartesianGrid(orig){
#ifdef LIBKEDF_DEBUG
    cout << "DEBUG: constructor (II) for OOP grid called." << endl;
#endif
    
    _realInSync = true;
    _reciInSync = true;
    _realReturned = true;
    _reciReturned = true;
    
    // allocate SIMD-enabled scratch space
    const size_t totalDimsReci = _xRecDim*_yRecDim*_zRecDim;
    const size_t totalDimsReal = _xDim*_yDim*_zDim;
    
#ifdef LIBKEDF_NO_ALIGNED_MEMORY
    _rawMemReal = (double*) fftw_malloc(totalDimsReal*sizeof(double));
    _rawMemReci = (fftw_complex*) fftw_malloc(totalDimsReci*sizeof(fftw_complex));
#else
    const size_t alignment = LIBKEDF_ALIGNMENT;
    const int error1 = posix_memalign((void**)&_rawMemReal,alignment,totalDimsReal*sizeof(double));
    const int error2 = posix_memalign((void**)&_rawMemReci,alignment,totalDimsReci*sizeof(complex<double>));
    if(error1 != 0 || error2 != 0){
        cerr << "ERROR: Can't get aligned host memory: " << error1 << " " << error2 << ", size: " << totalDimsReal << " " << totalDimsReci << ", alignment: " << alignment << endl;
        throw runtime_error("No aligned memory could be allocated.");
    }
#endif
    
    // re-planing for dimensions planed for before can be assumed to be fast
    _planR2C = fftw_plan_dft_r2c_3d(_zDim, _yDim, _xDim, _rawMemReal, _rawMemReci, FFTW_MEASURE | FFTW_DESTROY_INPUT);
    _planC2R = fftw_plan_dft_c2r_3d(_zDim, _yDim, _xDim, _rawMemReci, _rawMemReal, FFTW_MEASURE | FFTW_DESTROY_INPUT);
    
    _realGrid = new cube(_rawMemReal,_xDim,_yDim,_zDim,false,true);
    
    // recast the FFTW type
    complex<double>* tmp = reinterpret_cast<complex<double>*>(_rawMemReci);
    
    _reciGrid = new cx_cube(tmp,_xRecDim,_yRecDim,_zRecDim,false,true);
}

CartesianOOPGrid::~CartesianOOPGrid(){
#ifdef LIBKEDF_DEBUG
    cout << "DEBUG: destructor for OOP grid called." << endl;
#endif
        
#ifdef LIBKEDF_NO_ALIGNED_MEMORY
    fftw_free(_rawMemReci);
    fftw_free(_rawMemReal);
#else
    free(_rawMemReal);
    free(_rawMemReci);
#endif
    fftw_destroy_plan(_planR2C);
    fftw_destroy_plan(_planC2R);
    delete _realGrid;
    delete _reciGrid;
}

unique_ptr<CartesianOOPGrid> CartesianOOPGrid::duplicate() const{
    
#ifdef LIBKEDF_DEBUG
    cout << "DEBUG: creating CartesianOOPGrid duplicate of OOP grid." << endl;
#endif
    
    unique_ptr<CartesianOOPGrid> dup(new CartesianOOPGrid(*this));
    
    const size_t totalDimsReci = _xRecDim*_yRecDim*_zRecDim;
    const size_t totalDimsReal = _xDim*_yDim*_zDim;
    
    // only copy the state if it is in sync (i.e., if it is worth having)
    if(_realInSync){
#ifdef _OPENMP
#ifdef LIBKEDF_NO_ALIGNED_MEMORY
        #pragma omp parallel for default(none) shared(dup)
#else
        #pragma omp parallel for simd default(none) shared(dup)
#endif
        for(size_t x = 0; x < totalDimsReal; ++x){
            dup->_rawMemReal[x]= _rawMemReal[x];
        }
#else
        memcpy(dup->_rawMemReal,_rawMemReal,totalDimsReal*sizeof(double));
#endif
    }
    if(_reciInSync){
#ifdef _OPENMP
#ifdef LIBKEDF_NO_ALIGNED_MEMORY
        #pragma omp parallel for default(none) shared(dup)
#else
        #pragma omp parallel for simd default(none) shared(dup)
#endif
        for(size_t x = 0; x < totalDimsReci; ++x){
            dup->_rawMemReci[x][0] = _rawMemReci[x][0];
            dup->_rawMemReci[x][1] = _rawMemReci[x][1];
        }
#else
        memcpy(dup->_rawMemReci,_rawMemReci,totalDimsReci*sizeof(fftw_complex));
#endif
    }
    
    dup->_realInSync = _realInSync;
    dup->_reciInSync = _reciInSync;
    
    return dup;
}

unique_ptr<CartesianOOPGrid> CartesianOOPGrid::emptyDuplicate() const{
    
#ifdef LIBKEDF_DEBUG
    cout << "DEBUG: creating empty CartesianOOPGrid duplicate of OOP grid." << endl;
#endif
    
    unique_ptr<CartesianOOPGrid> dup(new CartesianOOPGrid(*this));
    // even though there is just garbage in the allocation, we define them to be in sync as to avoid FFT-ing garbage when we request either to write in
    dup->_realInSync = true;
    dup->_reciInSync = true;
    
    return dup;
}

void CartesianOOPGrid::addGrid(CartesianOOPGrid *other){
    
#ifdef LIBKEDF_DEBUG
    cout << "DEBUG: addGrid in CartesianOOPGrid called." << endl;
#endif
    
    cube* myGrid = this->getRealGrid();
    const cube* otherGrid = other->readRealGrid();
  
    const uword elems = myGrid->n_elem;
    
#ifdef LIBKEDF_NO_ALIGNED_MEMORY
    #pragma omp parallel for default(none) shared(myGrid,otherGrid)
#else
    #pragma omp parallel for simd default(none) shared(myGrid,otherGrid)
#endif
    for(uword x = 0; x < elems; ++x){
        myGrid->at(x) += otherGrid->at(x);
    }
    this->complete(myGrid);
}

void CartesianOOPGrid::fmaGrid(const double mult, CartesianOOPGrid *other){
    
#ifdef LIBKEDF_DEBUG
    cout << "DEBUG: fma in CartesianOOPGrid called." << endl;
#endif
    
    if(abs(mult) < Grid::NUMERICALACCURACY){
        return;
    }
    if(abs(1.0-mult) < Grid::NUMERICALACCURACY){
        addGrid(other);
        return;
    }
    
    cube* myGrid = this->getRealGrid();
    const cube* otherGrid = other->readRealGrid();
    
    const uword elems = myGrid->n_elem;
    
#ifdef LIBKEDF_NO_ALIGNED_MEMORY
    #pragma omp parallel for default(none) shared(myGrid,otherGrid)
#else
    #pragma omp parallel for simd default(none) shared(myGrid,otherGrid)
#endif
    for(uword x = 0; x < elems; ++x){
        myGrid->at(x) += mult*otherGrid->at(x);
    }
    this->complete(myGrid);
}

void CartesianOOPGrid::multiplyTwoSqrtOf(CartesianOOPGrid* other){
    
    const cube* otherGrid = other->readRealGrid();
    cube* myGrid = this->getRealGrid();
    
    const uword elems = myGrid->n_elem;
    
    #pragma omp parallel for default(none) shared(myGrid,otherGrid)
    for(uword x = 0; x < elems; ++x){
        myGrid->at(x) *= 2*sqrt(otherGrid->at(x));
    }
    this->complete(myGrid);
}

void CartesianOOPGrid::copyStateIn(const CartesianOOPGrid * fromGrid){
    
#ifdef LIBKEDF_DEBUG
    cout << "DEBUG: copying state in from OOP grid. This function is unsafe and only for internal use!" << endl;
#endif

    const size_t totalDimsReci = _xRecDim*_yRecDim*_zRecDim;
    const size_t totalDimsReal = _xDim*_yDim*_zDim;
    
    const size_t totalDimsReciOther = fromGrid->_xRecDim*fromGrid->_yRecDim*fromGrid->_zRecDim;
    const size_t totalDimsRealOther = fromGrid->_xDim*fromGrid->_yDim*fromGrid->_zDim;
    
    // some VERY BASIC sanity check, NO GUARANTEES THAT THE GRIDS ARE SIMILAR ENOUGH FROM THIS!
    if(totalDimsReci != totalDimsReciOther || totalDimsReal != totalDimsRealOther){
        cerr << "ERROR: Copying state in not allowed, grid dimensions different: " << totalDimsReci << "\t" << totalDimsReciOther << "\t" << totalDimsReal << "\t" << totalDimsRealOther << endl;
        throw runtime_error("Copying state in not allowed, grids are different!");
    }
    
    // now copy in the actual state
    if(fromGrid->_realInSync){
#ifdef _OPENMP
#ifdef LIBKEDF_NO_ALIGNED_MEMORY
        #pragma omp parallel for default(none) shared(fromGrid)
#else
        #pragma omp parallel for simd default(none) shared(fromGrid)
#endif
        for(size_t x = 0; x < totalDimsReal; ++x){
            _rawMemReal[x]= fromGrid->_rawMemReal[x];
        }
#else
        memcpy(_rawMemReal,fromGrid->_rawMemReal,totalDimsReal*sizeof(double));
#endif
    }
    if(fromGrid->_reciInSync){
#ifdef _OPENMP
#ifdef LIBKEDF_NO_ALIGNED_MEMORY
        #pragma omp parallel for default(none) shared(fromGrid)
#else
        #pragma omp parallel for simd default(none) shared(fromGrid)
#endif
        for(size_t x = 0; x < totalDimsReci; ++x){
            _rawMemReci[x][0] = fromGrid->_rawMemReci[x][0];
            _rawMemReci[x][1] = fromGrid->_rawMemReci[x][1];
        }
#else
        memcpy(_rawMemReci,fromGrid->_rawMemReci,totalDimsReci*sizeof(fftw_complex));
#endif
    }
    
    _realInSync = fromGrid->_realInSync;
    _reciInSync = fromGrid->_reciInSync;    
}

unique_ptr<FourierGrid> CartesianOOPGrid::createFourierDuplicate() const{
    
#ifdef LIBKEDF_DEBUG
    cout << "DEBUG: creating FourierGrid duplicate of OOP grid." << endl;
#endif
    
    unique_ptr<FourierGrid> dupUp = this->duplicate();
    
    return dupUp;
}

unique_ptr<FourierGrid> CartesianOOPGrid::createFourierEmptyDuplicate() const{
    
#ifdef LIBKEDF_DEBUG
        cout << "DEBUG: creating empty FourierGrid duplicate of OOP grid." << endl;
#endif
    
    unique_ptr<FourierGrid> dup = this->emptyDuplicate();
    
    return dup;
}
    
cube* CartesianOOPGrid::getRealGrid(){
    
    if(!_reciReturned){
        throw runtime_error("reciprocal grid must be returned before getting real grid");
    }
    if(!_realReturned){
        throw runtime_error("real grid must be returned before getting real grid");
    }
    
    if(!_realInSync){
#ifdef LIBKEDF_DEBUG
        cout << "DEBUG: doing c2r FFT in OOP grid." << endl;
#endif
        
        // do the FFT if we are not yet in the space.
        fftw_execute(_planC2R);        
    }
    
    _realReturned = false;
    _realInSync = true;
    _reciInSync = false; // due to DESTROY_INPUT && write access
    
    return _realGrid;
}

const cube* CartesianOOPGrid::tryReadRealGrid() const {
    
    if(!_reciReturned || !_realReturned || !_realInSync){
#ifdef LIBKEDF_DEBUG
        cout << "returning NULL grid..." << _reciReturned << _realReturned << _realInSync  << endl;
#endif
        return NULL;
    }

#ifdef LIBKEDF_DEBUG    
    cout << "returning real grid..." << endl;
#endif
        
    return _realGrid;
}

const cube* CartesianOOPGrid::readRealGrid() {
    
    getRealGrid();
    _realInSync = true; // read-only access
    _realReturned = true; // XXX: is this always true?
    
    return _realGrid;
}

cx_cube* CartesianOOPGrid::getReciprocalGrid(){
    
    if(!_realReturned){
        throw runtime_error("real grid must be returned before getting reciprocal grid");
    }
    if(!_reciReturned){
        throw runtime_error("reciprocal grid must be returned before getting reciprocal grid");
    }
    
    
    if(!_reciInSync){
#ifdef LIBKEDF_DEBUG
        cout << "DEBUG: doing r2c FFT in OOP grid." << endl;
#endif
        // do the FFT if we are not yet in the space.
        fftw_execute(_planR2C);
        // norm
        
        const uword elems = _reciGrid->n_elem;
#ifdef LIBKEDF_NO_ALIGNED_MEMORY
        #pragma omp parallel for default(none)
#else
        #pragma omp parallel for simd default(none)
#endif
        for(uword x = 0; x < elems; ++x){
             _reciGrid->at(x) /= _norm;
        }
    }
    
    _reciReturned = false;
    _reciInSync = true;
    _realInSync = false; // due to DESTROY_INPUT && write access
        
    return _reciGrid;
}

const cx_cube* CartesianOOPGrid::readReciprocalGrid() {
    
    getReciprocalGrid();
    _reciInSync = true; // read-only access
    _reciReturned = true; // XXX: is this always true?
    
    return _reciGrid;
}

void CartesianOOPGrid::complete(cube* realGrid){
    
    if(realGrid == NULL){
        throw runtime_error("real grid null");
    }
    
    // compare addresses
    if(realGrid != _realGrid){
        // problem
        throw runtime_error("real grid pointer returned not equal to real grid pointer given.");
    }
    
    _realReturned = true;
    
    realGrid = NULL;
}

void CartesianOOPGrid::completeReciprocal(cx_cube* reciprocalGrid){
    
    if(reciprocalGrid == NULL){
        throw runtime_error("reciprocal grid null");
    }
    
    // compare addresses
    if(reciprocalGrid != _reciGrid){
        // problem
        throw runtime_error("reciprocal grid pointer returned not equal to reciprocal grid pointer given.");
    }
    
    _reciReturned = true;
    
    reciprocalGrid = NULL;
}

void CartesianOOPGrid::resetToReal(){
    
    if(!_reciReturned || !_realReturned){
        throw runtime_error("must return all grid pointers before reset.");
    }
    
    // everything is in sync
    _reciInSync = true;
    _realInSync = true;
}

void CartesianOOPGrid::resetToReciprocal(){
    
    if(!_reciReturned || !_realReturned){
        throw runtime_error("must return all grid pointers before reset.");
    }
    
    // everything is in sync
    _reciInSync = true;
    _realInSync = true;
}
