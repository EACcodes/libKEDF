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
#include <iostream>
#include <fstream>
#include <string>
#include <stdexcept>
#ifdef __GNUC__
#include <sstream>
#endif
#include "Configuration.hpp"
#include "StringUtils.hpp"

Configuration::Configuration(string configFile){
    
    this->_cellVectors = make_shared<mat>(3,3);
    
    string line;
    ifstream file(configFile);
    if (file.is_open()) {
        while (getline(file, line)) {
            StringUtils::trim(line);
            if(line.compare("GRID FILE") == 0){
                getline(file,line);
                StringUtils::trim(line);
                _gridFile = line;
            } else if(line.compare("GRID CONFIG") == 0){
                getline(file,line);
                StringUtils::trim(line);
                _gridConfig = line;
            } else if(line.compare("KEDF CONFIG") == 0){
                getline(file,line);
                StringUtils::trim(line);
                _kedfConfig = line;
            } else if(line.compare("JOB TYPE") == 0){
                getline(file,line);
                StringUtils::trim(line);
                if(line.compare("energy") == 0){
                    _job = ENERGY;
                } else if(line.compare("potential") == 0){
                    _job = POTENTIAL;
                } else if(line.compare("stress") == 0){
                    _job = STRESS;
                } else {
                    throw runtime_error("Illegal job type " + line);
                }
            } else if(line.compare("ITERATIONS") == 0){
                getline(file,line);
                StringUtils::trim(line);
#ifdef __GNUC__
                stringstream ss(line);
                ss >> _noIterations;
#else
                _noIterations = stoi(line);
#endif
            } else if(line.compare("CARTESIAN GRID DIMENSIONS") == 0){
                getline(file,line);
                StringUtils::trim(line);
#ifdef __GNUC__
                stringstream ss(line);
                ss >> _xDim;
                ss >> _yDim;
                ss >> _zDim;
#else
                const vector<string> spl = StringUtils::split(line);
                if(spl.size() != 3){
                    throw runtime_error("There must be three Cartesian grid dimensions.");
                }
                _xDim = stoi(spl[0]);
                _yDim = stoi(spl[1]);
                _zDim = stoi(spl[2]);
#endif
            } else if(line.compare("CARTESIAN GRID LENGTHS") == 0){
                
                // here, we assume a "simple" grid. I.e., all cell angles are 90 deg
                _cellVectors->fill(0.0);
                double lengthX, lengthY, lengthZ;
                
                getline(file,line);
                StringUtils::trim(line);
#ifdef __GNUC__
                stringstream ss(line);
                ss >> lengthX;
                ss >> lengthY;
                ss >> lengthZ;
                
                lengthX *= 1.889725989;
                lengthY *= 1.889725989;
                lengthZ *= 1.889725989;
#else
                const vector<string> spl = StringUtils::split(line);
                if(spl.size() != 3){
                    throw runtime_error("There must be three Cartesian grid lengths.");
                }
                // convert to a.u.
                lengthX = stod(spl[0])*1.889725989;
                lengthY = stod(spl[1])*1.889725989;
                lengthZ = stod(spl[2])*1.889725989;
#endif
                _cellVectors->at(0,0) = lengthX;
                _cellVectors->at(1,1) = lengthY;
                _cellVectors->at(2,2) = lengthZ;
                
            } else if(line.compare("CARTESIAN CELL VECTORS") == 0){
                
                double compX, compY, compZ;
                
                for(size_t i = 0; i < 3; ++i){
                    getline(file,line);
                    StringUtils::trim(line);
#ifdef __GNUC__
                    stringstream ss(line);
                    ss >> compX;
                    ss >> compY;
                    ss >> compZ;
                
                    compX *= 1.889725989;
                    compY *= 1.889725989;
                    compZ *= 1.889725989;
#else
                    const vector<string> spl = StringUtils::split(line);
                    if(spl.size() != 3){
                        throw runtime_error("There must be three Cartesian vector components.");
                    }
                    // convert to a.u.
                    compX = stod(spl[0])*1.889725989;
                    compY = stod(spl[1])*1.889725989;
                    compZ = stod(spl[2])*1.889725989;
#endif
                    _cellVectors->at(i,0) = compX;
                    _cellVectors->at(i,1) = compY;
                    _cellVectors->at(i,2) = compZ;
                }
                
            } else if(line.compare("FILL GRID WITH ZEROS") == 0){
                _fillStyle = ZEROS;
            } else if(line.compare("FILL GRID RANDOM") == 0){
                _fillStyle = RANDOM;
            } else if(line.compare("FILL GRID FROM FILE") == 0){
                _fillStyle = FROMFILE;
            } else if(line.compare("PRINT VERBOSE") == 0){
                _printVerbose = true;
            } else {
                throw runtime_error("Illegal input " + line);
            }
        }
        file.close();
    } else {
        throw runtime_error("Unable to open file " + configFile + " for reading configuration in!");
    }
    
}

Configuration::~Configuration(){
}

int Configuration::getNoIterations() const {
    return _noIterations;
}

string Configuration::getGridFile() const {
    return _gridFile;
}

string Configuration::getKEDFConfig() const {
    return _kedfConfig;
}

string Configuration::getGridConfig() const {
    return _gridConfig;
}

job_types_t Configuration::getJobType() const {
    return _job;
}

size_t Configuration::getXDim() const {
    return _xDim;
}

size_t Configuration::getYDim() const {
    return _yDim;
}

size_t Configuration::getZDim() const {
    return _zDim;
}

shared_ptr<mat> Configuration::getCellVectors() const {
    return _cellVectors;
}

fillstyle_t Configuration::fillStyle() const {
    return _fillStyle;
}

bool Configuration::printVerbose() const {
    return _printVerbose;
}
