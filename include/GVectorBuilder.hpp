/* 
 * Author: Johannes M Dieterich
 */

#ifndef GVECTORBUILDER_HPP
#define	GVECTORBUILDER_HPP

#include <armadillo>
#include <memory>
using namespace std;

class GVectorBuilder{

public:
    static void buildGVectors(const shared_ptr<arma::mat> cellVectors,
            const size_t recX, const size_t recY, const size_t recZ,
            shared_ptr<arma::cube> & gNorms, shared_ptr<arma::cube>& gVectorsX,
            shared_ptr<arma::cube>& gVectorsY, shared_ptr<arma::cube>& gVectorsZ);
    
    static void buildGVectors(const shared_ptr<arma::mat> cellVectors,
            const size_t recX, const size_t recY, const size_t recZ,
            shared_ptr<arma::fcube>& gNorms, shared_ptr<arma::fcube>& gVectorsX,
            shared_ptr<arma::fcube>& gVectorsY, shared_ptr<arma::fcube>& gVectorsZ);

    static void buildGVectors(const shared_ptr<arma::mat> cellVectors,
            const size_t recX, const size_t recY, const size_t recZ,
            const size_t offX, const size_t offY, const size_t offZ,
            const size_t endX, const size_t endY, const size_t endZ,
            shared_ptr<arma::cube> & gNorms, shared_ptr<arma::cube>& gVectorsX,
            shared_ptr<arma::cube>& gVectorsY, shared_ptr<arma::cube>& gVectorsZ);

    static void buildGVectors(const shared_ptr<arma::mat> cellVectors,
            const size_t recX, const size_t recY, const size_t recZ,
            const size_t offX, const size_t offY, const size_t offZ,
            const size_t endX, const size_t endY, const size_t endZ,
            shared_ptr<arma::fcube>& gNorms, shared_ptr<arma::fcube>& gVectorsX,
            shared_ptr<arma::fcube>& gVectorsY, shared_ptr<arma::fcube>& gVectorsZ);

    
private:
    const static bool DEBUG = false;
};

#endif	/* GVECTORBUILDER_HPP */

