/* 
 * Copyright (c) 2016-2017, Princeton University, Johannes M Dieterich, Emily A Carter
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

#ifndef LIBKEDF_H
#define LIBKEDF_H

typedef enum { SMP, OCL, OCLSP, MPIIP } gridtype;

#ifdef __cplusplus

#include "KEDF.hpp"
#include "CartesianOOPGrid.hpp"
#ifdef LIBKEDF_OCL
#include "CartesianOCLOOPGrid.hpp"
#endif

struct libkedf_data {
    gridtype configuredGridType = SMP;
    
    KEDF<CartesianOOPGrid>* kedf;
    CartesianOOPGrid* grid;
    CartesianOOPGrid* potential;
    
#ifdef LIBKEDF_OCL
    KEDF<CartesianOCLOOPGrid>* kedfOCL;
    CartesianOCLOOPGrid* gridOCL;
    CartesianOCLOOPGrid* potentialOCL;
#endif
};

#else

struct libkedf_data {
    gridtype configuredGridType;

    void* kedf;
    void* grid;
    void* potential;

#ifdef LIBKEDF_OCL
    void* kedfOCL;
    void* gridOCL;
    void* potentialOCL;
#endif
};

#endif

#ifdef __cplusplus
extern "C" {
#endif

    /**
     * Initialize the necessary auxiliary data structure for libKEDF. Must be called
     * prior to any other calls!
     * @return dat pointer to the data structure. Changed on exit.
     */
    struct libkedf_data * libkedf_init_();
    
    /**
     * Initialize a regular Cartesian shared-memory grid (potentially with OpenMP). One of the grid initializations must be called before initializing a KEDF.
     * @param dat pointer to the data structure. Content changed on exit.
     * @param x pointer to the number of grid points in x (first) dimension. Must be larger than 0.
     * @param y pointer to the number of grid points in y (second) dimension. Must be larger than 0.
     * @param z pointer to the number of grid points in z (third) dimension. Must be larger than 0.
     * @param vecX the unit cell vector in x dimension. Must contain three Cartesian components.
     * @param vecY the unit cell vector in y dimension. Must contain three Cartesian components.
     * @param vecZ the unit cell vector in z dimension. Must contain three Cartesian components.
     */
    void libkedf_initialize_grid_(struct libkedf_data *dat, const int *x, const int *y, const int *z,
        const double *vecX, const double *vecY, const double *vecZ);
    
#ifdef LIBKEDF_OCL
    /**
     * Initialize an OpenCL-accelerated Cartesian SMP grid. One of the grid initializations must be called before initializing a KEDF.
     * @param dat pointer to the data structure. Content changed on exit.
     * @param x pointer to the number of grid points in x (first) dimension. Must be larger than 0.
     * @param y pointer to the number of grid points in y (second) dimension. Must be larger than 0.
     * @param z pointer to the number of grid points in z (third) dimension. Must be larger than 0.
     * @param vecX the unit cell vector in x dimension. Must contain three Cartesian components.
     * @param vecY the unit cell vector in y dimension. Must contain three Cartesian components.
     * @param vecZ the unit cell vector in z dimension. Must contain three Cartesian components.
     * @param platformNo pointer to the OpenCL platform to be used. Must be 0 or larger.
     * @param deviceNo pointer to the device number to be used within this platform. Must be 0 or larger.
     */
    void libkedf_initialize_grid_ocl_(struct libkedf_data *dat, const int *x, const int *y, const int *z,
        const double *vecX, const double *vecY, const double *vecZ, const int *platformNo,
        const int *deviceNo);
#endif
    
    /**
     * Update the cell vectors in use by this libKEDF interface.
     * @param dat pointer to the data structure. Content changed on exit.
     * @param vecX the new unit cell vector in x dimension. Must contain three Cartesian components.
     * @param vecY the new unit cell vector in y dimension. Must contain three Cartesian components.
     * @param vecZ the new unit cell vector in z dimension. Must contain three Cartesian components.
     */
    void libkedf_update_cellvectors_(struct libkedf_data *dat, const double *vecX, const double *vecY, const double *vecZ);
    
    /**
     * Initialize a Thomas-Fermi KEDF.
     * @param dat pointer to the data structure. Content changed on exit.
     */
    void libkedf_initialize_tf_(struct libkedf_data *dat);
    
    /**
     * Initialize a von Weizsaecker KEDF.
     * @param dat pointer to the data structure. Content changed on exit.
     */
    void libkedf_initialize_vw_(struct libkedf_data *dat);
    
    /**
     * Initialize a compound a*TF + b*vW KEDF.
     * @param dat pointer to the data structure. Content changed on exit.
     * @param a pointer to the factor a for the Thomas-Fermi contribution
     * @param b pointer to the factor b for the von Weizsaecker contribution
     */
    void libkedf_initialize_tf_plus_vw_(struct libkedf_data *dat, const double *a, const double *b);

    /**
     * Initialize a Wang-Teter KEDF (including TF and vW contributions) with original Wang-Teter settings.
     * @param dat pointer to the data structure. Content changed on exit.
     * @param rho0 pointer to the uniform density to be used in the WT kernel.
     */
    void libkedf_initialize_wt_(struct libkedf_data *dat, const double *rho0);
    
    /**
     * Initialize a Wang-Teter KEDF (including TF and vW contributions) with custom settings.
     * @param dat pointer to the data structure. Content changed on exit.
     * @param rho0 pointer to the uniform density to be used in the WT kernel.
     * @param alpha pointer to the alpha exponent for the density in the nonlocal term
     * @param beta pointer to the beta exponent for the density in the nonlocal term
     */
    void libkedf_initialize_wt_custom_(struct libkedf_data *dat, const double *rho0, const double *alpha, const double *beta);
    
    /**
     * Initialize a Wang-Teter KEDF (including TF and vW contributions) with Smargiassi-Madden settings.
     * @param dat pointer to the data structure. Content changed on exit.
     * @param rho0 pointer to the uniform density to be used in the WT kernel.
     */
    void libkedf_initialize_sm_(struct libkedf_data *dat, const double *rho0);
    
    /**
     * Initialize a 1st-order Taylor-expanded Wang-Govind-Carter KEDF (including TF and vW contributions) with default settings.
     * @param dat pointer to the data structure. Content changed on exit.
     * @param rho0 pointer to the uniform density to be used in the WGC kernel.
     */
    void libkedf_initialize_wgc1st_(struct libkedf_data *dat, const double *rho0);
    
    /**
     * Initialize a 1st-order Taylor-expanded Wang-Govind-Carter KEDF (including TF and vW contributions) with custom settings.
     * @param dat pointer to the data structure. Content changed on exit.
     * @param rho0 pointer to the uniform density to be used in the WGC kernel.
     * @param alpha pointer to the alpha exponent for the density in the nonlocal term
     * @param beta pointer to the beta exponent for the density in the nonlocal term
     * @param gamma pointer to the gamma value in the WGC kernel
     */
    void libkedf_initialize_wgc1st_custom_(struct libkedf_data *dat, const double *rho0, const double *alpha, const double *beta, const double *gamma);

    /**
     * Initialize a 2nd-order Taylor-expanded Wang-Govind-Carter KEDF (including TF and vW contributions) with default settings.
     * @param dat pointer to the data structure. Content changed on exit.
     * @param rho0 pointer to the uniform density to be used in the WGC kernel.
     */
    void libkedf_initialize_wgc2nd_(struct libkedf_data *dat, const double *rho0);
    
    /**
     * Initialize a 2nd-order Taylor-expanded Wang-Govind-Carter KEDF (including TF and vW contributions) with custom settings.
     * @param dat pointer to the data structure. Content changed on exit.
     * @param rho0 pointer to the uniform density to be used in the WGC kernel.
     * @param alpha pointer to the alpha exponent for the density in the nonlocal term
     * @param beta pointer to the beta exponent for the density in the nonlocal term
     * @param gamma pointer to the gamma value in the WGC kernel
     */
    void libkedf_initialize_wgc2nd_custom_(struct libkedf_data *dat, const double *rho0, const double *alpha, const double *beta, const double *gamma);
    
    /**
     * Initialize a Huang-Carter KEDF (including TF and vW contributions) with default settings.
     * @param dat pointer to the data structure. Content changed on exit.
     * @param rho0 pointer to the uniform density to be used in the HC kernel.
     * @param lambda pointer to the lambda mixing factor in HC
     */
    void libkedf_initialize_hc_(struct libkedf_data *dat, const double *rho0, const double *lambda);
    
    /**
     * Initialize a Huang-Carter KEDF (including TF and vW contributions) with custom settings.
     * @param dat pointer to the data structure. Content changed on exit.
     * @param rho0 pointer to the uniform density to be used in the HC kernel.
     * @param lambda pointer to the lambda mixing factor in HC
     * @param alpha pointer to the alpha exponent
     * @param beta pointer to the beta exponent
     * @param refRatio pointer to the reference ratio for density binning
     */
    void libkedf_initialize_hc_custom_(struct libkedf_data *dat, const double *rho0, const double *lambda, const double *alpha, const double *beta, const double *refRatio);
    
    /**
     * Calculate the kinetic energy from a given electronic density.
     * @param dat pointer to the data structure. Content changed on exit.
     * @param density the electronic density. Array of values in column-major order, size of array must correspond to number of grid points needed. I.e., for a simple Cartesian grid: #x * #y * #z.
     * @param energy pointer to the kinetic energy value. Value changed on exit.
     */
    void libkedf_energy_(struct libkedf_data *dat, const double* density, double *energy);
    
    /**
     * Calculate the kinetic energy and potential thereof from a given electronic density.
     * @param dat pointer to the data structure. Content changed on exit.
     * @param density the electronic density. Array of values in column-major order, size of array must correspond to number of grid points needed. I.e., for a simple Cartesian grid: #x * #y * #z.
     * @param potential the potential. Array of values in column-major order, size of array must correspond to number of grid points needed. I.e., for a simple Cartesian grid: #x * #y * #z. Values changed on exit.
     * @param energy pointer to the kinetic energy value. Value changed on exit.
     */
    void libkedf_potential_(struct libkedf_data *dat, const double* density, double* potential, double *energy);
    
    /**
     * Clean, free, and deallocate auxiliary data structures. Must be called after all libKEDF calls have happened (typically at end of program execution).
     * @param dat pointer to the data structure. Changed on exit.
     */
    void libkedf_cleanup_(struct libkedf_data *dat);
    
#ifdef __cplusplus
}
#endif

#endif /* LIBKEDF_H */

