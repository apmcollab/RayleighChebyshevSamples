#include <iostream>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <algorithm>
//
// LaplaceOp2dEigTest_FD.cpp
//
// A test code that demonstrates the use of the RayleighChebyshev procedure
// to determine a specified number of eigenvectors and eigenvalues of
// the five point discretization to the Laplace operator with
// homogeneous Dirichlet boundary conditions. 
//
// The operator is implemented as a finite difference operator, 
// e.g. the operator is not represented as a matrix and the vector space 
// is the set of "2D" vectors associated with the values of the 
// nodes of a uniform 2D rectangular grid. 
//
// Note: The finite difference operator is defined at all points
// of the rectangular grid and sets boundary values to 0.
// To avoid computing the 0 eigenvalues associated with the operator 
// action on the boundary, the method implicitly carries out the iteration 
// on a subspace that is orthogonal to the subspace that spans the
// boundary values. In order to do this, both the discrete Laplace
// operator AND the randomize operator must set the boundary values
// to zero after each invocation. 
// 
//
// The compilation of this program does not require a Lapack installation. 
//
// Directory set up for this test program
//
//             [Samples Directory]
// -------------------------------------------------
//         |                     |                
//   [LaplaceOp2dEigTest]   [Components]
//                               |
//                      -------------------
//                      Component directories
//
// where the compondent directories contain the source directories (already included when initializing submodules):
//
//    DoubleVectorNd    (https://github.com/apmcollab/SCC-DoubleVectorNd)
//    GridFunctionNd    (https://github.com/apmcollab/SCC-GridFunctionNd)
//    RandOpNd          (https://github.com/apmcollab/SCC-RandOpNd)
//    RayleighChebyshev (https://github.com/apmcollab/RayleighChebyshev)
//
// The command line compilation command is
//
// g++ LaplaceOp2dEigTest.cpp -std=c++17 -D RC_WITHOUT_LAPACK_ -I../Components   -o LaplaceOp2dEigTest.exe
//
// To enable OpenMP add -fopenmp to the above command line (or the equivalent flag for your compiler).
//
// To build and run the test program using CMake, cd to the LaplaceOp2dEigTest directory and then execute 
//
// mkdir build
// cd build
// cmake ../
// make release
// ctest -V
//
// The CMakeLists.txt file distributed does not toggle on OpenMP, to use OpenMP, edit the line in the CMakeLists.txt file  
//
// OPTION(USE_OPENMP  "Option USE_OPENMP"  OFF) 
//  to be 
// OPTION(USE_OPENMP  "Option USE_OPENMP"  ON)
//
// and then rerun the build steps above. 
//
//
// Oct. 10, 2024
//


#include "RayleighChebyshev/RayleighChebyshev.h"

// Vector class used in test

#include "GridFunctionNd/SCC_GridFunction2d.h"
#include "GridFunctionNd/SCC_GridFunction2dUtility.h"

// Operator that creates instances of vector class with random entries

#include "RandOpNd/RandOpDirichlet2d.h"

// Operator class used in test

#include "LaplaceOp2d.h"

#ifdef _OPENMP
#include <omp.h>
#endif

int main()
{

// OpenMP setup

#ifdef _OPENMP
    int threadCount = 4;
    if(threadCount > omp_get_max_threads()){threadCount = omp_get_max_threads();}
    if(threadCount <= 0)                   {threadCount = omp_get_max_threads();}
    omp_set_num_threads(threadCount);

    printf("\n");
    printf("#############\n");
    printf("############# Using OpenMP With %d Threads\n",omp_get_max_threads());
    printf("#############\n");
    printf("\n");
#endif


//   Problem set up.

	double xMin = 0.0;
	double xMax = 1.0;

	double yMin = 0.0;
	double yMax = 1.0;

	long xPanels = 20;
	long yPanels = 20;

	double LX   = (xMax-xMin);
	double hx   = (xMax-xMin)/(double)(xPanels);

	double LY   = (yMax-yMin);
	double hy   = (yMax-yMin)/(double)(yPanels);

	long nx     = xPanels - 1;
	long ny     = yPanels - 1;

	double pi     =  3.141592653589793238;
	double alpha  = -1.0;  // Coefficient of discrete Laplace operator

    // Compute exact eigenvalues of the discrete operator knowing that 
    // the eigenvectors of the discrete operator are the dicrete sin functions
    //
    // e.g. the functions 
    //
    // sin(k1*pi*(x-xMin)/(xMax-xMin))*sin(k2*pi*(y-yMin)/(yMax-yMin))
    // 
    // evaluated at the grid points of the uniform rectangular grid.
    //

    std::vector<double> exactEigValues(nx*ny,0.0);

    long index = 0;
	for(long k1 = 1; k1 <= nx; k1++)
	{
	for(long k2 = 1; k2 <= ny; k2++)
	{
	exactEigValues[index] = alpha*(
			(2.0/(hx*hx))*(std::cos((k1*pi*hx)/LX) - 1.0)
		  + (2.0/(hy*hy))*(std::cos((k2*pi*hy)/LY) - 1.0));
    index++;
	}}

    // Create sorted list of algebraically smallest to algebraically largest

    std::sort(exactEigValues.begin(),exactEigValues.end());

    // Instantiate Laplace Operator

    LaplaceOp2d Lop2d(alpha);

    // Instantiate Random Operator

    SCC::RandOpDirichlet2d  randomOp;

    // Allocate arrays for eigenvectors and eigenvalues

    std::vector <SCC::GridFunction2d>  eigVectors;
    std::vector <double>                  eigValues;

    // Declare an instance of the Raylegh-Chebyshev eigensystem procedure

    RayleighChebyshev < SCC::GridFunction2d, LaplaceOp2d , SCC::RandOpDirichlet2d > RCeigProcedure;
    //                         |                |                       |
    //                         |                |                       |
    //                 vector class    linear operator class     randomize operator class

    RCeigProcedure.setEigDiagnosticsFlag(true);
    RCeigProcedure.setVerboseFlag(true);

    SCC::GridFunction2d vTmp(xPanels,xMin,xMax,yPanels,yMin,yMax);     // A temporary vector is required as input. This vector must
                                                                       // be a non-null instance of the vector class

    double dimension           = vTmp.getDimension();   // The dimension of the vector space.
    double subspaceTol         = 2.0e-6;                // The stopping tolerance.
    long subspaceIncrementSize = 10;                    // Subspace size used to determine the eigenpairs.
    long bufferSize            = 3;                     // The number of additional eigenpairs that are internally computed but not output.
    
                                                        //
                                                        // The size (or dimension) of the buffer subspace must be sufficiently large  
                                                        // so there is at least one eigenpair in that subspace whose eigenvalue is distinct from 
                                                        // the set of eigenvalues in the subspace containing the desired eigenpairs.
                                                        //
                                                        // 
    //                                                  
    // Specifying the number of eigenpairs to compute. Typically one chooses subspaceIncrementSize to 
    // be equal to the desired number of eigenpairs. If subspaceIncrementSize is less than eigCount, then 
    // subspaceIncrementSize eigenpairs are computed incrementally until the total number of eigenpairs 
    // are computed. 
    //                                                  
    long eigCount              = dimension < 10 ? dimension : 10;
    
    printf("\n\nXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n\n");
    
    printf("XXXX   LaplaceOp2dEig_Test Results XXXX\n\n");
    printf("XXXX   Operator implemented as finite difference operator XXXX\n");
    printf("XXXX   Using default parameters\n");
    printf("XXXX   Tolerance Specified : %10.5e\n\n",subspaceTol);
    
    printf("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n\n");
    

    RCeigProcedure.getMinEigenSystem(eigCount, subspaceTol, subspaceIncrementSize, bufferSize, vTmp,
    		Lop2d, randomOp, eigValues, eigVectors);

    printf("       Eigenvalue              Error       Relative Error \n");
    for(long  k = 0; k < eigCount; k++ )
    {
    	printf("%-5ld %-20.14e  %10.5e   %-10.5e\n", k+1, 
    	eigValues[k], std::abs(eigValues[k] - exactEigValues[k]),std::abs(eigValues[k] -exactEigValues[k])/std::abs(exactEigValues[k]));
    }
    
    //
    // Check error in the first distinct computed eigenvector (k1 = 1 and k2=1) 
    //

    SCC::GridFunction2d exactEigVector(xPanels,xMin,xMax,yPanels,yMin,yMax);
    
    double x; 
    double y; 
    for(long i = 0; i <= xPanels; i++)
    {
    for(long j = 0; j <= yPanels; j++)
    {
    x = xMin + i*hx;
    y = yMin + j*hy;
    exactEigVector(i,j) = std::sin((pi*(x-xMin))/(xMax-xMin))*std::sin((pi*(y-yMin))/(yMax-yMin));
    }}
    
    // normalize 
    
    exactEigVector *= 1.0/exactEigVector.norm2();
    
    // fix up the sign if necessary 

    if(exactEigVector.dot(eigVectors[0]) < 0) {exactEigVector *= -1.0;} 
    
    // Compute the error 
    
    SCC::GridFunction2d eigVecError =  exactEigVector;
    eigVecError -= eigVectors[0];
    
    double eigVecErrorL2   = eigVecError.norm2();
    double eigVecErrorInf = eigVecError.normInf();
    
    
    printf("\n\nEigenvector (0,0) error (L2)  : %10.5e \n",eigVecErrorL2);
    printf("Eigenvector (0,0) error (Inf) : %10.5e \n",eigVecErrorInf);
   
    // Sample run again with alternate parameter settings
    
    // setStopCondition values 
    //
    // One of  DEFAULT, COMBINATION, RESIDUAL_ONLY, EIGENVALUE_ONLY  
    //
    // EIGENVALUE_ONLY : Only monitors the convergence of the eigenvalues 
    //                   Eigenvalues typically converge more quickly than 
    //                   the eigenvectors, so with this stopping condition
    //                   the resulting eigenvectors may have reduced accuracy.
    //                   Based upon perturbation theory the expected accuracy 
    //                   in the eigenvectors will be the square root of the 
    //                   eigenvalue accuracy. If one wants to insure that this 
    //                   eigenvector accuracy expectation is met, then 
    //                   specify COMBINATION instead. 
    //                   
    //
    // RESIDUAL_ONLY   : Only monitors the convergence of the eigenvectors
    //                   Numerical precision may limit the obtainable residual,
    //                   if this value is too small, the method may not converge.
    //    
    // COMBINATION     : Monitor convergence of both eigenvalues and eigenvectors
    //                   and terminate the iteration when the eigenvalue stopping
    //                   condition is less than the subspaceTol and the eigenvector
    //                   stopping condition is less than the square root of the 
    //                   subspaceTol.
    //                   
    //
    // DEFAULT         : Currently equal to COMBINATION
    
    std::string stopCondition = "RESIDUAL_ONLY";
    RCeigProcedure.setStopCondition(stopCondition);
   
    
    dimension             = vTmp.getDimension();   
    subspaceTol           = 2.0e-4;              // Sets the stopping tolerance for the eigenvectors 
                                                 // when RESIDUAL_ONLY is specified for stop condition            
    subspaceIncrementSize = 10;                    
    bufferSize            = 3;                                                                  
    eigCount              = dimension < 10 ? dimension : 10;

    // Remove previous results 
    
    eigValues.clear();
    eigVectors.clear();
    
    printf("\n\nXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n\n");
    
    printf("XXXX   LaplaceOp2dEig_Test Results XXXX\n\n");
    printf("XXXX   Operator implemented as finite difference operator XXXX\n");
    printf("XXXX   Using residual only stopping condition \n");
    printf("XXXX   Tolerance Specified : %10.5e\n\n",subspaceTol);
    
    printf("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n\n");
    
    RCeigProcedure.getMinEigenSystem(eigCount, subspaceTol, subspaceIncrementSize, bufferSize, vTmp,
    		Lop2d, randomOp, eigValues, eigVectors);



    printf("       Eigenvalue              Error       Relative Error \n");
    
    for(long  k = 0; k < eigCount; k++ )
    {
    	printf("%-5ld %-20.14e  %10.5e   %-10.5e\n", k+1, 
    	eigValues[k], std::abs(eigValues[k] - exactEigValues[k]),std::abs(eigValues[k] -exactEigValues[k])/std::abs(exactEigValues[k]));
    }
    
    //
    // Check error in the first distinct computed eigenvector (k1 = 1 and k2=1) 
    //
    
    for(long i = 0; i <= xPanels; i++)
    {
    for(long j = 0; j <= yPanels; j++)
    {
    x = xMin + i*hx;
    y = yMin + j*hy;
    exactEigVector(i,j) = std::sin((pi*(x-xMin))/(xMax-xMin))*std::sin((pi*(y-yMin))/(yMax-yMin));
    }}
    
    // normalize 
    
    exactEigVector *= 1.0/exactEigVector.norm2();
    
    // fix up the sign if necessary 

    if(exactEigVector.dot(eigVectors[0]) < 0) {exactEigVector *= -1.0;} 
    
    // Compute the error 
    
    eigVecError =  exactEigVector;
    eigVecError -= eigVectors[0];
    
    eigVecErrorL2   = eigVecError.norm2();
    eigVecErrorInf = eigVecError.normInf();
    
    
    printf("\n\nEigenvector (0,0) error (L2)  : %10.5e \n",eigVecErrorL2);
    printf("Eigenvector (0,0) error (Inf) : %10.5e \n",eigVecErrorInf);
	printf("\nXXX Execution Completed XXXX\n");
	return 0;

}

