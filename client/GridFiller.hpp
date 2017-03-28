/* 
 * Author: Johannes M Dieterich
 */

#ifndef GRIDFILLER_HPP
#define	GRIDFILLER_HPP

#include "Grid.hpp"

class GridFiller {
    
public:
    static void fillGrid(Grid* grid, const std::string fileName);
    
    static void fillEmptyGrid(Grid* grid);
    
    static void fillGridRandomly(Grid* grid);
};

#endif	/* GRIDFILLER_HPP */

