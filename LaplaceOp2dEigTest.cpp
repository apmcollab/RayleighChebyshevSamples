#include <iostream>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <algorithm>
//
// LaplaceOp2dEigTest.cpp
//
// A test code that demonstrates the use of the RayleighChebyshev procedure
// to determine a specified number of eigenvectors and eigenvalues of
// the five point discretization to the Laplace operator with
// homogeneous Dirichlet boundary conditions. 
//
// Directory set up for this test program
//
// [Test directory]
// ---------------------------
//         |                |                
//   [TestProgram]     Components
//                          |
//                   -------------------
//                 Component directories
//
// where the compondent directories contain the source directories:
//
//    DoubleVectorNd
//    GridFunctionNd
//    RandOpNd
//    RayleighChebyshev
//
// The command line compilation command is
//
// g++ LaplaceOp2dEigTest.cpp -std=c++17 -D RC_WITHOUT_LAPACK_ -I../Components   -o LaplaceOp2dEigTest.exe
//
// To enable OpenMP add -fopenmp to the above command line (or the equivalent flag for your compiler).
//
// Oct. 10, 2024
//

// Method

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
    int threadCount = -1;
    cout << "Enter in number of threads: ";
    cin >> threadCount;

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
			(2.0/(hx*hx))*(cos((k1*pi*hx)/LX) - 1.0)
		  + (2.0/(hy*hy))*(cos((k2*pi*hy)/LY) - 1.0));
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

    RCeigProcedure.setEigDiagnosticsFlag();
    RCeigProcedure.setVerboseFlag();

    SCC::GridFunction2d vTmp(xPanels,xMin,xMax,yPanels,yMin,yMax);     // A temporary vector is required as input. This vector must
                                                                         // be a non-null instance of the vector class

    double dimension           = vTmp.getDimension();
    double subspaceTol         = 2.0e-6;
    long subspaceIncrementSize = 10;
    long bufferSize            = 3;
    long eigCount              = dimension < 10 ? dimension : 10;

    RCeigProcedure.getMinEigenSystem(eigCount, subspaceTol, subspaceIncrementSize, bufferSize, vTmp,
    		Lop2d, randomOp, eigValues, eigVectors);

    printf("\n\nXXXX   RC_OperatorEig_Test Results XXXX\n\n");
    printf("XXXX   Using default parameters\n");
    printf("XXXX   Tolerance Specified : %10.5e\n\n",subspaceTol);

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
   
    // Sample 

	printf("\nXXX Execution Completed XXXX\n");
	return 0;

}

