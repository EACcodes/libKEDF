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
#include <armadillo>
#include <iostream>
#include <fstream>
#include "GridFiller.hpp"
#include "StringUtils.hpp"
using namespace std;
#ifdef __GNUC__
#include <sstream>
#endif
using namespace arma;

void GridFiller::fillGrid(Grid* grid, const string fileName){

    cube* gr = grid->getRealGrid();
    const size_t dimX = gr->n_rows;
    const size_t dimY = gr->n_cols;
    const size_t dimZ = gr->n_slices;
    
    const size_t totElem = dimX*dimY*dimZ;
    
    size_t c = 0;
    
    string line;
    ifstream file(fileName);
    if (file.is_open()) {
        while (getline(file, line)) {
            const vector<string> spl = StringUtils::split(line);
            for(size_t x = 0; x < spl.size(); ++x){
#ifdef __GNUC__
                stringstream ss(line);
                double d;
                ss >> d;
                gr->at(c) = d;
#else
                gr->at(c) = stod(spl[x]);
#endif
                ++c;
            }
        }
        file.close();
    } else {
        throw runtime_error("Unable to open file " + fileName + " for reading grid in!");
    }
    
    if(c != totElem){
        cerr << "total elements " << totElem << endl;
        cerr << "read elements " << c << endl;
        throw runtime_error("Not the right number of elements got read.");
    }
    
    grid->complete(gr);
}

void GridFiller::fillEmptyGrid(Grid* grid){
    
    cube* gr = grid->getRealGrid();
    gr->fill(0.0);
    grid->complete(gr);
}

void GridFiller::fillGridRandomly(Grid* grid){
    
    cube* gr = grid->getRealGrid();
    gr->randu();
    grid->complete(gr);
}