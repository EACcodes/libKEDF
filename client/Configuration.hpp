/* 
 * Author: Johannes M Dieterich
 */

#ifndef CONFIGURATION_HPP
#define	CONFIGURATION_HPP

#include <armadillo>
#include <memory>
#include <string>
using namespace arma;
using namespace std;

enum job_types_t {ENERGY, POTENTIAL, STRESS};
enum fillstyle_t {ZEROS, RANDOM, FROMFILE};

class Configuration {
public:
    
    Configuration(string configFile);
    ~Configuration();
    
    int getNoIterations() const;
    
    string getGridFile() const;
    
    string getKEDFConfig() const;
    
    string getGridConfig() const;
    
    job_types_t getJobType() const;
    
    size_t getXDim() const;
    
    size_t getYDim() const;
    
    size_t getZDim() const;
    
    shared_ptr<mat> getCellVectors() const;
    
    fillstyle_t fillStyle() const;
    
    bool printVerbose() const;
    
private:
    int _noIterations = 100;
    size_t _xDim = 0;
    size_t _yDim = 0;
    size_t _zDim = 0;
    shared_ptr<mat> _cellVectors;
    string _gridFile = "density.grid";
    string _kedfConfig = "vonWeizsaecker";
    string _gridConfig = "fftw3,out-of-place";
    fillstyle_t _fillStyle = ZEROS;
    bool _printVerbose = false;
    job_types_t _job = ENERGY;
};

#endif	/* CONFIGURATION_HPP */

