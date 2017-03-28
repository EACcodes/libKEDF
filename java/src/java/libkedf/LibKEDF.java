/*
Copyright (c) 2015-2016, Princeton University, Johannes M Dieterich, Emily A Carter
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software without
   specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
package libkedf;

import java.io.Serializable;

/**
 * An interface to libKEDF. We recommend constructing with the LibKEDFBuilder.
 * Currently NOT thread-safe. Please create multiple interface objects for this
 * application.
 * @author Johannes M DIeterich
 * @version 2016-10-26
 */
public class LibKEDF implements Serializable, Cloneable {
    
    private static final long serialVersionUID = (long) 20161024;
    
    public static enum GRIDTYPE {REGULAR, OPENCL};
    public static enum KEDF {ThomasFermi, vonWeizsaecker, aTFPlusbVW, WangTeter, WangGovindCarter1st, WangGovindCarter2nd, HuangCarter};
    
    private final long myHandle;
    
    private final int gridX;
    private final int gridY;
    private final int gridZ;
    
    /**
     * Constructor to build a Thomas-Fermi or von Weizsaecker KEDF.
     * @param gridType the type of grid. Must not be null.
     * @param kedf the KEDF to be constructed. Must not be null and must be either Thomas-Fermi or von Weizsaecker.
     * @param gridConf the grid configuration
     * @throws Exception if something is not done/configured right
     */
    public LibKEDF(final GRIDTYPE gridType, final KEDF kedf, final GridConfig gridConf) throws Exception {
        this(gridType,kedf,gridConf.gridPointsX, gridConf.gridPointsY, gridConf.gridPointsZ,
                gridConf.vecX,gridConf.vecY,gridConf.vecZ,0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, gridConf.platform, gridConf.device);
    }
    
    /**
     * Constructor to build a TF + vW KEDF.
     * @param gridType the type of grid. Must not be null.
     * @param gridConf the grid configuration
     * @param aFactor the a factor
     * @param bFactor the b factor
     * @throws Exception if something is not done/configured right
     */
    public LibKEDF(final GRIDTYPE gridType, final GridConfig gridConf, final double aFactor, final double bFactor) throws Exception {
        this(gridType,KEDF.aTFPlusbVW,gridConf.gridPointsX, gridConf.gridPointsY, gridConf.gridPointsZ,
                gridConf.vecX,gridConf.vecY,gridConf.vecZ,aFactor, bFactor,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                gridConf.platform, gridConf.device);
    }
    
    /**
     * Constructor for Wang-Teter KEDFs.
     * @param gridType the grid type. Must not be null.
     * @param kedf the KEDF choice. Must be Wang-Teter.
     * @param gridConf the grid configuration
     * @param rho0WT  the rho0 value for this system
     * @param alphaWT the alpha factor for WT
     * @param betaWT the beta factor for WT
     * @throws Exception if something is not done/configured right
     */
    public LibKEDF(final GRIDTYPE gridType, final KEDF kedf, final GridConfig gridConf, final double rho0WT, final double alphaWT, final double betaWT) throws Exception {
        this(gridType,KEDF.WangTeter,gridConf.gridPointsX, gridConf.gridPointsY, gridConf.gridPointsZ,
                gridConf.vecX,gridConf.vecY,gridConf.vecZ, 0.0, 0.0, rho0WT, alphaWT, betaWT, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, gridConf.platform, gridConf.device);
        if(kedf != KEDF.WangTeter){
            throw new RuntimeException("Constructor must only be used for Wang-Teter KEDF. Not for " + kedf.name());
        }
    }
    
    /**
     * Constructor to build a Taylor-expanded WGC KEDF.
     * @param gridType the grid type. Must not be null.
     * @param kedf the KEDF choice. Must be either 1st or 2nd order Taylor-expanded WGC.
     * @param gridConf the grid configuration
     * @param rho0WGC the rho0 value for this system
     * @param alphaWGC the alpha value for WGC
     * @param betaWGC the beta value for WGC
     * @param gammaWGC the gamma value for WGC
     * @throws Exception if something is not done/configured right
     */
    public LibKEDF(final GRIDTYPE gridType, final KEDF kedf, final GridConfig gridConf,
            final double rho0WGC, final double alphaWGC, final double betaWGC,
            final double gammaWGC) throws Exception {
        this(gridType,kedf,gridConf.gridPointsX, gridConf.gridPointsY, gridConf.gridPointsZ,
                gridConf.vecX,gridConf.vecY,gridConf.vecZ, 0.0, 0.0, 0.0, 0.0, 0.0,
                rho0WGC, alphaWGC, betaWGC, gammaWGC, 0.0, 0.0, 0.0, 0.0, 0.0,
                gridConf.platform, gridConf.device);
    }
    
    /**
     * Constructor to build a HC KEDF.
     * @param gridType the type of grid. Must not be null.
     * @param gridConf the grid configuration
     * @param rho0HC the rho0 value for this system
     * @param lambdaHC the lambda value for HC
     * @param alphaHC the alpha value for HC
     * @param betaHC the beta value for HC
     * @param refRatioHC the reference ratio for HC
     * @throws Exception if something is not done/configured right
     */
    public LibKEDF(final GRIDTYPE gridType, final GridConfig gridConf, final double rho0HC,
            final double lambdaHC, final double alphaHC, final double betaHC, final double refRatioHC) throws Exception {
        this(gridType,KEDF.HuangCarter,gridConf.gridPointsX, gridConf.gridPointsY,
                gridConf.gridPointsZ, gridConf.vecX,gridConf.vecY,gridConf.vecZ,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, rho0HC, lambdaHC,
                alphaHC, betaHC, refRatioHC, gridConf.platform, gridConf.device);
    }
    
    private LibKEDF(final GRIDTYPE gridType, final KEDF kedf, final int gridX, final int gridY, final int gridZ,
            final double[] unitX, final double[] unitY, final double[] unitZ, final double aFactorTF, final double bFactorVW,
            final double rho0WT, final double alphaWT, final double betaWT, final double rho0WGC, final double alphaWGC,
            final double betaWGC, final double gammaWGC, final double rhoHC, final double lambdaHC, final double alphaHC,
            final double betaHC, final double refRatioHC, final int platform, final int device) throws Exception {
        
        assert(gridType != null);
        assert(kedf != null);
        assert(gridX > 0);
        assert(gridY > 0);
        assert(gridZ > 0);
        
        this.gridX = gridX;
        this.gridY = gridY;
        this.gridZ = gridZ;
        
        // first initialize the grid
        if(gridType == GRIDTYPE.REGULAR){
            myHandle = initializeRegularGrid(gridX, gridY, gridZ, unitX, unitY, unitZ);
        } else {
            myHandle = initializeOpenCLGrid(gridX, gridY, gridZ, unitX, unitY, unitZ, platform, device);
        }
        
        // now initialize the KEDF
        switch(kedf){
            case ThomasFermi:
                initializeTF(myHandle);
                break;
            case vonWeizsaecker:
                initializeVW(myHandle);
                break;
            case aTFPlusbVW:
                initializeATFBVW(myHandle, aFactorTF, bFactorVW);
                break;
            case WangTeter:
                initializeWT(myHandle, rho0WT, alphaWT, betaWT);
                break;
            case WangGovindCarter1st:
                initializeTayloredWGC(myHandle, 1, rho0WGC, alphaWGC, betaWGC, gammaWGC);
                break;
            case WangGovindCarter2nd:
                initializeTayloredWGC(myHandle, 2, rho0WGC, alphaWGC, betaWGC, gammaWGC);
                break;
            case HuangCarter:
                initializeHC(myHandle, rhoHC, alphaHC, betaHC, lambdaHC, refRatioHC);
                break;
        }
    }
    
    public LibKEDF(final LibKEDF orig){
        assert(orig != null);
        this.myHandle = orig.myHandle;
        this.gridX = orig.gridX;
        this.gridY = orig.gridY;
        this.gridZ = orig.gridZ;
    }
    
    @Override
    public LibKEDF clone(){
        return new LibKEDF(this);
    }
    
    public double energy(final double[] rho){
        if(rho.length != gridX*gridY*gridZ){
            throw new RuntimeException("Density handed over is not of the correct dimension (" + gridX + " x " + gridY + " x " + gridZ + ").");
        }
        
        return calcEnergy(myHandle, rho);
    }
    
    public double potential(final double[] rho, final double[] potential){
        if(rho.length != gridX*gridY*gridZ){
            throw new RuntimeException("Density handed over is not of the correct dimension (" + gridX + " x " + gridY + " x " + gridZ + ").");
        }
        if(potential.length != gridX*gridY*gridZ){
            throw new RuntimeException("Potential handed over is not of the correct dimension (" + gridX + " x " + gridY + " x " + gridZ + ").");
        }
        
        return calcPotential(myHandle, rho, potential);
    }
    
    public void cleanup(){
        cleanup(myHandle);
    }
   
    static { System.loadLibrary("KEDFjni"); } // we DO NOT use the regular libKEDF but the KEDFjni library that has a thin shim to translate JVM types
    
    private static native long initializeRegularGrid(final int gridX, final int gridY, final int gridZ, final double[] unitX, final double[] unitY, final double[] unitZ);
    
    private static native long initializeOpenCLGrid(final int gridX, final int gridY, final int gridZ, final double[] unitX, final double[] unitY, final double[] unitZ, final int platform, final int device);
    
    private static native void initializeTF(final long handle);
    
    private static native void initializeVW(final long handle);
    
    private static native void initializeATFBVW(final long handle, final double a, final double b);
    
    private static native void initializeWT(final long handle, final double rho0, final double alpha, final double beta);
    
    private static native void initializeTayloredWGC(final long handle, final int order, final double rho0, final double alpha, final double beta, final double gamma);
    
    private static native void initializeHC(final long handle, final double rho0, final double alpha, final double beta, final double lambda, final double refRatioHC);

    private static native double calcEnergy(final long handle, final double[] density);
    
    private static native double calcPotential(final long handle, final double[] density, final double[] potential);
    
    private static native void cleanup(final long handle);
}
