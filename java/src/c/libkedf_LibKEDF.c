#include "libkedf_LibKEDF.h"
#include "libKEDF.h"

JNIEXPORT jlong JNICALL Java_libkedf_LibKEDF_initializeRegularGrid
  (JNIEnv * env, jclass class, jint gridX, jint gridY, jint gridZ, jdoubleArray vecX, jdoubleArray vecY, jdoubleArray vecZ){
    struct libkedf_data* auxData = libkedf_init_();
    
    const int x = (int) gridX;
    const int y = (int) gridY;
    const int z = (int) gridZ;
    
    double* xCell = (*env)->GetDoubleArrayElements(env, vecX, 0);
    double* yCell = (*env)->GetDoubleArrayElements(env, vecY, 0);
    double* zCell = (*env)->GetDoubleArrayElements(env, vecZ, 0);
    
    libkedf_initialize_grid_(auxData, &x, &y, &z, xCell, yCell, zCell);
    
    (*env)->ReleaseDoubleArrayElements(env, vecX, xCell, 0);
    (*env)->ReleaseDoubleArrayElements(env, vecY, yCell, 0);
    (*env)->ReleaseDoubleArrayElements(env, vecZ, zCell, 0);
    
    return (jlong) auxData;
}

#ifdef LIBKEDF_OCL
JNIEXPORT jlong JNICALL Java_libkedf_LibKEDF_initializeOpenCLGrid
  (JNIEnv * env, jclass class, jint gridX, jint gridY, jint gridZ, jdoubleArray vecX, jdoubleArray vecY, jdoubleArray vecZ,
        jint platform, jint device){
    struct libkedf_data* auxData = libkedf_init_();
    
    const int x = (int) gridX;
    const int y = (int) gridY;
    const int z = (int) gridZ;
    
    double* xCell = (*env)->GetDoubleArrayElements(env, vecX, 0);
    double* yCell = (*env)->GetDoubleArrayElements(env, vecY, 0);
    double* zCell = (*env)->GetDoubleArrayElements(env, vecZ, 0);
    
    const int plat = (int) platform;
    const int dev = (int) device;
    
    libkedf_initialize_grid_ocl_(auxData, &x, &y, &z, xCell, yCell, zCell, &plat, &dev);
    
    (*env)->ReleaseDoubleArrayElements(env, vecX, xCell, 0);
    (*env)->ReleaseDoubleArrayElements(env, vecY, yCell, 0);
    (*env)->ReleaseDoubleArrayElements(env, vecZ, zCell, 0);
    
    return (jlong) auxData;
}
#endif

JNIEXPORT void JNICALL Java_libkedf_LibKEDF_initializeTF
  (JNIEnv * env, jclass class, jlong auxPointer){
    
    struct libkedf_data *auxData = (struct libkedf_data*) auxPointer;
    libkedf_initialize_tf_(auxData);
}

JNIEXPORT void JNICALL Java_libkedf_LibKEDF_initializeVW
  (JNIEnv * env, jclass class, jlong auxPointer){
    struct libkedf_data *auxData = (struct libkedf_data*) auxPointer;
    libkedf_initialize_vw_(auxData);
}

JNIEXPORT void JNICALL Java_libkedf_LibKEDF_initializeATFBVW
  (JNIEnv * env, jclass class, jlong auxPointer, jdouble a, jdouble b){
    struct libkedf_data *auxData = (struct libkedf_data*) auxPointer;
    const double aFac = (double) a;
    const double bFac = (double) b;
    libkedf_initialize_tf_plus_vw_(auxData, &aFac, &bFac);
}

JNIEXPORT void JNICALL Java_libkedf_LibKEDF_initializeWT
  (JNIEnv * env, jclass class, jlong auxPointer, jdouble rho0, jdouble alpha, jdouble beta){
    struct libkedf_data *auxData = (struct libkedf_data*) auxPointer;
    const double rho0Fac = (double) rho0;
    const double alphaFac = (double) alpha;
    const double betaFac = (double) beta;
    libkedf_initialize_wt_custom_(auxData, &rho0Fac, &alphaFac, &betaFac);
}

JNIEXPORT void JNICALL Java_libkedf_LibKEDF_initializeTayloredWGC
  (JNIEnv * env, jclass class, jlong auxPointer, jint order, jdouble rho0, jdouble alpha, jdouble beta, jdouble gamma){
    struct libkedf_data *auxData = (struct libkedf_data*) auxPointer;
    const double rho0Fac = (double) rho0;
    const double alphaFac = (double) alpha;
    const double betaFac = (double) beta;
    const double gammaFac = (double) gamma;
    const int wgcOrder = (int) order;
    if(wgcOrder == 1){
        libkedf_initialize_wgc1st_custom_(auxData, &rho0Fac, &alphaFac, &betaFac, &gammaFac);
    } else if(wgcOrder == 2){
        libkedf_initialize_wgc2nd_custom_(auxData, &rho0Fac, &alphaFac, &betaFac, &gammaFac);
    } else {
        printf("Order %d not allowed in Taylor-expanded WGC. \n", order);
    }
}

JNIEXPORT void JNICALL Java_libkedf_LibKEDF_initializeHC
  (JNIEnv * env, jclass class, jlong auxPointer, jdouble rho0, jdouble alpha, jdouble beta, jdouble lambda, jdouble refRatio){
    struct libkedf_data *auxData = (struct libkedf_data*) auxPointer;
    const double rho0Fac = (double) rho0;
    const double alphaFac = (double) alpha;
    const double betaFac = (double) beta;
    const double lambdaFac = (double) lambda;
    const double refRatioFac = (double) refRatio;
    libkedf_initialize_hc_custom_(auxData, &rho0Fac, &lambdaFac, &alphaFac, &betaFac, &refRatioFac);
}

JNIEXPORT jdouble JNICALL Java_libkedf_LibKEDF_calcEnergy
  (JNIEnv * env, jclass class, jlong auxPointer, jdoubleArray density){
    struct libkedf_data *auxData = (struct libkedf_data*) auxPointer;
    double e = -42;
    double* densPtr = (*env)->GetDoubleArrayElements(env, density, 0);
    libkedf_energy_(auxData, densPtr, &e);
    (*env)->ReleaseDoubleArrayElements(env, density, densPtr, 0);
    
    return e;
}

JNIEXPORT jdouble JNICALL Java_libkedf_LibKEDF_calcPotential
  (JNIEnv * env, jclass class, jlong auxPointer, jdoubleArray density, jdoubleArray potential){
    struct libkedf_data *auxData = (struct libkedf_data*) auxPointer;
    double e = -42;
    double* densPtr = (*env)->GetDoubleArrayElements(env, density, 0);
    double* potPtr = (*env)->GetDoubleArrayElements(env, potential, 0);
    libkedf_potential_(auxData, densPtr, potPtr, &e);
    (*env)->ReleaseDoubleArrayElements(env, density, densPtr, 0);
    (*env)->ReleaseDoubleArrayElements(env, potential, potPtr, 0);
    
    return e;
}

JNIEXPORT void JNICALL Java_libkedf_LibKEDF_cleanup
  (JNIEnv * env, jclass class, jlong auxPointer){
    struct libkedf_data *auxData = (struct libkedf_data*) auxPointer;
    libkedf_cleanup_(auxData);
}

