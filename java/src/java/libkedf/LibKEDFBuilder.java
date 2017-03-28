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

/**
 * A builder for the libKEDF interface.
 * @author Johannes M Dieterich
 * @version 2016-10-30
 */
public class LibKEDFBuilder {

    private LibKEDF.GRIDTYPE grid = LibKEDF.GRIDTYPE.REGULAR;
    private LibKEDF.KEDF kedf = LibKEDF.KEDF.ThomasFermi;
    
    // for a*TF + b*vW
    private double aFactorTF = 1.0;
    private double bFactorVW = 1.0;
    
    // for WT / SM / "old" WGC / ...
    private double rho0WT = -42; // this must be configured if WT is used!
    private double alphaWT = 5.0/6.0;
    private double betaWT = 5.0/6.0;
    
    // for WGC99
    private double rho0WGC = -42; // this must be configured if WGC99 is used!
    private double alphaWGC = 1.20601132958330;
    private double betaWGC = 0.460655337083368;
    private double gammaWGC = 2.7;
    
    // for HC10
    private double rho0HC = -42; // this must be configured if HC10 is used!
    private double lambdaHC = -42; // this must be configured if HC10 is used!
    private double alphaHC = 2.0166666666666666;
    private double betaHC = 0.65;
    private double refRatioHC = 1.15;
    
    // grid dimensions
    private final int gridX;
    private final int gridY;
    private final int gridZ;
    
    // unit cell dimensions
    private final double[] unitCellX;
    private final double[] unitCellY;
    private final double[] unitCellZ;
    
    private int platform = 0;
    private int device = 0;
    
    /**
     * Default constructor required to set the absolute minimum of information.
     * @param unitCellX the Cartesian unit cell vector in x. Must have 3 elements.
     * @param unitCellY the Cartesian unit cell vector in y. Must have 3 elements.
     * @param unitCellZ the Cartesian unit cell vector in z. Must have 3 elements.
     * @param gridX the number of grid points in x dimension. Must be larger than 0.
     * @param gridY the number of grid points in y dimension. Must be larger than 0.
     * @param gridZ the number of grid points in z dimension. Must be larger than 0.
     * @throws java.lang.Exception if the handed over data is not correct
     */
    public LibKEDFBuilder(final double[] unitCellX, final double[] unitCellY, final double[] unitCellZ,
            final int gridX, final int gridY, final int gridZ) throws Exception {
        
        if(gridX <= 0 || gridY <= 0 || gridZ <= 0){
            throw new Exception("All grid dimensions must be positive. Are " + gridX + ", " + gridY + ", " + gridZ + ".");
        }
        
        if(unitCellX == null || unitCellX.length != 3){
            throw new Exception("Unit cell vector for x-dimension must not be null and must have exactly 3 elemensts.");
        }
        
        if(unitCellY == null || unitCellY.length != 3){
            throw new Exception("Unit cell vector for y-dimension must not be null and must have exactly 3 elemensts.");
        }
       
        if(unitCellZ == null || unitCellZ.length != 3){
            throw new Exception("Unit cell vector for z-dimension must not be null and must have exactly 3 elemensts.");
        }
        
        this.gridX = gridX;
        this.gridY = gridY;
        this.gridZ = gridZ;
        this.unitCellX = unitCellX.clone();
        this.unitCellY = unitCellY.clone();
        this.unitCellZ = unitCellZ.clone();
    }
    
    /**
     * Configure the grid type.
     * @param type the grid type. Must not be null.
     * @return this builder
     */
    public LibKEDFBuilder configGridType(final LibKEDF.GRIDTYPE type){
        assert(type != null);
        this.grid = type;
        
        return this;
    }
    
    /**
     * Configure the platform choice for an OpenCL grid.
     * @param platform the platform. Must be positive.
     * @return this builder
     */
    public LibKEDFBuilder configureOCLPlatform(final int platform){
        assert(platform > 0);
        this.platform = platform;
        
        return this;
    }
    
    /**
     * Configure the device choice for an OpenCL grid.
     * @param device the platform. Must be positive.
     * @return this builder
     */
    public LibKEDFBuilder configureOCLDevice(final int device){
        assert(device > 0);
        this.device = device;
        
        return this;
    }
    
    /**
     * Configure the KEDF to use.
     * @param kedf the KEDF to use. Must not be null.
     * @return this builder
     */
    public LibKEDFBuilder configKEDF(final LibKEDF.KEDF kedf){
        assert(kedf != null);
        this.kedf = kedf;
        
        return this;
    }
    
    /**
     * Configure the a in a*TF+b*vW
     * @param a the a parameter, should be larger than 0.0
     * @return this builder
     */
    public LibKEDFBuilder configureA_TFvW(final double a){
        assert(a >= 0.0);
        this.aFactorTF = a;
        
        return this;
    }
    
    /**
     * Configure the b in a*TF+b*vW
     * @param b the b parameter, should be larger than 0.0
     * @return this builder
     */
    public LibKEDFBuilder configureB_TFvW(final double b){
        assert(b >= 0.0);
        this.bFactorVW = b;
        
        return this;
    }
    
    /**
     * Configure the rho0 in WT
     * @param rho0 the rho0 parameter, should be larger than 0.0
     * @return this builder
     */
    public LibKEDFBuilder configureRho0WT(final double rho0){
        assert(rho0 > 0.0);
        this.rho0WT = rho0;
        
        return this;
    }
    
    /**
     * Configure the alpha in WT
     * @param alpha the alpha parameter, should be larger than 0.0
     * @return this builder
     */
    public LibKEDFBuilder configureAlphaWT(final double alpha){
        assert(alpha > 0.0);
        this.alphaWT = alpha;
        
        return this;
    }
    
    /**
     * Configure the beta in WT
     * @param beta the beta parameter, should be larger than 0.0
     * @return this builder
     */
    public LibKEDFBuilder configureBetaWT(final double beta){
        assert(beta > 0.0);
        this.betaWT = beta;
        
        return this;
    }
    
    /**
     * Configure the rho0 in WGC (1st or 2nd order)
     * @param rho0 the rho0, should be larger than 0.0
     * @return this builder
     */
    public LibKEDFBuilder configureRho0WGC(final double rho0){
        assert(rho0 > 0.0);
        this.rho0WGC = rho0;
        
        return this;
    }
    
    /**
     * Configure the alpha in WGC (1st or 2nd order)
     * @param alpha the alpha parameter, should be larger than 0.0
     * @return this builder
     */
    public LibKEDFBuilder configureAlphaWGC(final double alpha){
        assert(alpha > 0.0);
        this.alphaWGC = alpha;
        
        return this;
    }
    
    /**
     * Configure the beta in WGC (1st or 2nd order)
     * @param beta the beta parameter, should be larger than 0.0
     * @return this builder
     */
    public LibKEDFBuilder configureBetaWGC(final double beta){
        assert(beta > 0.0);
        this.betaWGC = beta;
        
        return this;
    }
    
    /**
     * Configure the gamma in WGC (1st or 2nd order)
     * @param gamma the gamma parameter, should be larger than 0.0
     * @return this builder
     */
    public LibKEDFBuilder configureGammaWGC(final double gamma){
        assert(gamma > 0.0);
        this.gammaWGC = gamma;
        
        return this;
    }
    
    /**
     * Configure the rho0 in HC
     * @param rho0 the rho0 parameter, should be larger than 0.0
     * @return this builder
     */
    public LibKEDFBuilder configureRho0HC(final double rho0){
        assert(rho0 > 0.0);
        this.rho0HC = rho0;
        
        return this;
    }
    
    /**
     * Configure the lambda in HC
     * @param lambda the lambda parameter, should be larger than 0.0
     * @return this builder
     */
    public LibKEDFBuilder configureLambdaHC(final double lambda){
        assert(lambda > 0.0);
        this.lambdaHC = lambda;
        
        return this;
    }
    
    /**
     * Configure the alpha in HC
     * @param alpha the alpha parameter, should be larger than 0.0
     * @return this builder
     */
    public LibKEDFBuilder configureAlphaHC(final double alpha){
        assert(alpha > 0.0);
        this.alphaHC = alpha;
        
        return this;
    }
    
    /**
     * Configure the beta in HC
     * @param beta the beta parameter, should be larger than 0.0
     * @return this builder
     */
    public LibKEDFBuilder configureBetaHC(final double beta){
        assert(beta > 0.0);
        this.betaHC = beta;
        
        return this;
    }
    
    /**
     * Configure the reference ratio in HC
     * @param refRatio the reference ratio parameter, should be larger than 0.0
     * @return this builder
     */
    public LibKEDFBuilder configureRefRatioHC(final double refRatio){
        assert(refRatio > 0.0);
        this.refRatioHC = refRatio;
        
        return this;
    }
    
    
    // couple of builder functions...
    
    public LibKEDF constructInterface() throws Exception {
        
        final GridConfig conf = new GridConfig();
        conf.gridPointsX = gridX;
        conf.gridPointsY = gridY;
        conf.gridPointsZ = gridZ;
        
        conf.vecX = unitCellX;
        conf.vecY = unitCellY;
        conf.vecZ = unitCellZ;
        
        conf.platform = platform;
        conf.device = device;
        
        switch(kedf){
            case ThomasFermi:
            case vonWeizsaecker:
                return new LibKEDF(grid, kedf, conf);
            case aTFPlusbVW:
                return new LibKEDF(grid, conf, aFactorTF, bFactorVW);
            case WangTeter:
                if(rho0WT <= 0){throw new RuntimeException("rho0 in WT must be larger than zero.");}
                return new LibKEDF(grid, kedf, conf, rho0WT, alphaWT, betaWT);
            case WangGovindCarter1st:
            case WangGovindCarter2nd:
                if(rho0WGC <= 0){throw new RuntimeException("rho0 in WGC must be larger than zero.");}
                return new LibKEDF(grid, kedf, conf, rho0WGC, alphaWGC, betaWGC, gammaWGC);
            case HuangCarter:
                if(rho0HC <= 0){throw new RuntimeException("rho0 in HC must be larger than zero.");}
                if(lambdaHC <= 0){throw new RuntimeException("lambda in HC must be larger than zero.");}
                return new LibKEDF(grid, conf, rho0HC,
                    lambdaHC, alphaHC, betaHC, refRatioHC);
        }
        
        throw new RuntimeException("Please add the KEDF enum you were trying to construct to the switch statement above.");
    }
}
