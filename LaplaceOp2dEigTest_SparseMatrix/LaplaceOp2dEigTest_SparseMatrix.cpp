//XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
// LaplaceOp2dEigTest_SparseMatrix.cpp
//XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
// Laplace Operator 2D Eigensystem Test : Sparse matrix operator implementation
//
// A test code that demonstrates the use of the RayleighChebyshev procedure
// to determine a specified number of eigenvalues and eigenvectors of
// the five point discretization to the Laplace operator with
// homogeneous Dirichlet boundary conditions. 
//
// The finite difference operator is implemented as a sparse matrix acting
// upon a vector space of dimension equal to the number of interior points
// of a 2D rectangular grid. 
//
// The vector class used in this program is SCC::DoubleArray1d
//
// Both the sparse matrix class, SCC::SparseOp, and the randomize operator
// class, SCC::RandVectorOp, are templated with respect to a vector class. 
// 
//
// The compilation of this program does not require a Lapack installation. 
//
// Directory set up for this test program
//
//             [Samples Directory]
// -------------------------------------------------
//         |                             |                
//[LaplaceOp2dEigTest_SparseMatrix]   [Components]
//                                       |
//                               -------------------
//                              Component directories
//
// where the compondent directories contain the source directories (already included when initializing submodules):
//
//    SparseOp          (https://github.com/apmcollab/SparseOp)_
//    DoubleVectorNd    (https://github.com/apmcollab/SCC-DoubleVectorNd)
//    RandOpNd          (https://github.com/apmcollab/SCC-RandOpNd)
//    RayleighChebyshev (https://github.com/apmcollab/RayleighChebyshev)
//
// 
// The command line compilation command is
//
// g++ LaplaceOp2dEigTest_SparseMatrix.cpp -std=c++17 -D RC_WITHOUT_LAPACK_ -I../Components  -o LaplaceOp2dEigTest_SparseMatrix.exe
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
// Reference:
//
//   Christopher R. Anderson, "A Rayleigh-Chebyshev procedure for finding
//   the smallest eigenvalues and associated eigenvectors of large sparse
//   Hermitian matrices" Journal of Computational Physics,
//   Volume 229 Issue 19, September, 2010.
//
// Created on: Oct 11, 2024
//      Author: anderson
//
//#############################################################################
//#
//# Copyright  2024 Chris Anderson
//#
//# This program is free software: you can redistribute it and/or modify
//# it under the terms of the Lesser GNU General Public License as published by
//# the Free Software Foundation, either version 3 of the License, or
//# (at your option) any later version.
//#
//# This program is distributed in the hope that it will be useful,
//# but WITHOUT ANY WARRANTY; without even the implied warranty of
//# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//# GNU General Public License for more details.
//#
//# For a copy of the GNU General Public License see
//# <http://www.gnu.org/licenses/>.
//#
//#############################################################################

#include <iostream>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <algorithm>

#include "RayleighChebyshev/RayleighChebyshev.h" // RayleighChebyshev class

#include "SparseOp/SCC_SparseOp.h"               // Sparse matrix class used in test

#include "DoubleVectorNd/SCC_DoubleVector1d.h"   // Vector class used in test

#include "RandOpNd/RandVectorOp.h"               // Randomize vector operator class

#include "SparseOp/SCC_IndexMap2d.h"             // Utility class for setting up sparse matrix representation of 
                                                 // 2D finite difference operator
            
// Prototype for routine that sets up sparse matrix representation of 2D finite difference
// operator. Implementation follows main(..)

void setUpSparseOp(long xPanels, double xMin, double xMax, 
                   long yPanels, double yMin, double yMax, double alpha,
                   SCC::SparseOp<SCC::DoubleVector1d>& sOp);


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

//////////////////////////////////////////////////////////////////////////////////////
//                         Test problem set up
//////////////////////////////////////////////////////////////////////////////////////

//  Specifying parameters for 2d grid and Laplace operator coefficient

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


    // Coefficient of discrete Laplace operator; using -1.0 so operator is postive definite. 
	
	double alpha  = -1.0;  

	
    // Compute exact eigenvalues of the discrete operator knowing that 
    // the eigenvectors of the discrete operator are the dicrete sin functions
    //
    // e.g. the functions 
    //
    // sin(k1*pi*(x-xMin)/(xMax-xMin))*sin(k2*pi*(y-yMin)/(yMax-yMin))
    // 
    // evaluated at the grid points of the uniform rectangular grid.
    //
	double pi     =  3.141592653589793238;
	
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
    
//////////////////////////////////////////////////////////////////////////////////////
//                          Operator set up.
//////////////////////////////////////////////////////////////////////////////////////

    // Instantiate instance of sparse matrix class 
    
    SCC::SparseOp<SCC::DoubleVector1d> sOp;
    //                    |
    //                    |
    // Vector class template argument - see SCC_SparseOp.h file 
    // for information on the requirements of the vector class.                                   
 
    // Invoke function that initializes the sparse matrix entries
    
	setUpSparseOp(xPanels, xMin, xMax, yPanels, yMin, yMax, alpha, sOp);

    // Instantiate random Operator

    SCC::RandVectorOp<SCC::DoubleVector1d> randomOp;
    //                      |
    //                      |
    // Vector class template argument - see RandVectorO.h file 
    // for information on the requirements of the vector class.    
     
    // Declare an instance of the Raylegh-Chebyshev eigensystem procedure

    RayleighChebyshev < SCC::DoubleVector1d, SCC::SparseOp<SCC::DoubleVector1d>, SCC::RandVectorOp<SCC::DoubleVector1d> > RCeigProcedure;
    //                         |                     |                                         |
    //                         |                     |                                         |
    //                 vector class          linear operator class                   randomize operator class


    // Set diagnostics output

    RCeigProcedure.setEigDiagnosticsFlag(true);
    RCeigProcedure.setVerboseFlag(true);

//////////////////////////////////////////////////////////////////////////////////////
//                       Input/Output set up
//////////////////////////////////////////////////////////////////////////////////////

   // Creating input parameters

    long  systemSize = sOp.getRowDimension();
    
    SCC::DoubleVector1d vTmp(systemSize);               // A temporary vector is required as input. This vector must
                                                        // be a non-null instance of the vector class.

    double dimension           = vTmp.getDimension();   // The dimension of the vector space.
    double subspaceTol         = 1.0e-6;                // The stopping tolerance.
    
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
    
    // Allocate vectors for output of eigenvectors and eigenvalues

    std::vector <SCC::DoubleVector1d>    eigVectors;
    std::vector <double>                  eigValues;

//////////////////////////////////////////////////////////////////////////////////////
//           Computation of the eigensystem and evaluation of the error
//////////////////////////////////////////////////////////////////////////////////////
    
    printf("\n\nXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n\n");
    
    printf("XXXX   LaplaceOp2dEigTest_SparseMatrix Results XXXX\n\n");
    printf("XXXX   Operator implemented as sparse matrix XXXX\n");
    printf("XXXX   Using default parameters\n");
    printf("XXXX   Tolerance specified : %10.5e\n\n",subspaceTol);
    printf("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n\n");
    

    RCeigProcedure.getMinEigenSystem(eigCount, subspaceTol, subspaceIncrementSize, bufferSize, vTmp,
    		       sOp, randomOp, eigValues, eigVectors);


    printf("\nXXXX   Eigenvalue errors using stopping condition DEFAULT = COMBINATION \n");
    printf("XXXX   Expected maximal error is tolerance specified =  %10.5e \n\n",subspaceTol);

    printf("       Eigenvalue              Error       Relative Error \n");
    
    for(long  k = 0; k < eigCount; k++ )
    {
    	printf("%-5ld %-20.14e  %10.5e   %-10.5e\n", k+1, 
    	eigValues[k], std::abs(eigValues[k] - exactEigValues[k]),std::abs(eigValues[k] -exactEigValues[k])/std::abs(exactEigValues[k]));
    }
    
    //
    // Check error in the second distinct computed eigenvector (k1 = 2 and k2 = 2)
    //
    
    SCC::IndexMap2d iMap(nx,ny); // Index mapping from 2D grid points to linear index 

    SCC::DoubleVector1d exactEigVector(systemSize);
    
    double x; 
    double y; 
    long   k;
    
    for(long i = 0; i < nx; i++)
    {
    for(long j = 0; j < ny; j++)
    {
    x = xMin + (i+1)*hx;
    y = yMin + (j+1)*hy;
    k = iMap.linearIndex(i,j);
    exactEigVector(k) = std::sin((2.0*pi*(x-xMin))/(xMax-xMin))*std::sin((2.0*pi*(y-yMin))/(yMax-yMin));
    }}
    
    // normalize 
    
    exactEigVector *= 1.0/exactEigVector.norm2();
    
    // fix up the sign if necessary 

    if(exactEigVector.dot(eigVectors[3]) < 0) {exactEigVector *= -1.0;}
    
    // Compute the error 
    
    SCC::DoubleVector1d eigVecError =  exactEigVector;
    eigVecError -= eigVectors[3];
    
    double eigVecErrorL2   = eigVecError.norm2();
    double eigVecErrorInf = eigVecError.normInf();
    
    printf("\nXXXX   Eigenvector errors using stopping condition DEFAULT = COMBINATION \n");
    printf("XXXX   Expected maximal error is sqrt(tolerance specified) =  %10.5e \n\n",std::sqrt(subspaceTol));

    printf("Eigenvector (2,2) error (L2)  : %10.5e \n",eigVecErrorL2);
    printf("Eigenvector (2,2) error (Inf) : %10.5e \n",eigVecErrorInf);
    
    
//////////////////////////////////////////////////////////////////////////////////////
//           Re-computation of the eigensystem and evaluation of the error
//           with stopping condition set to RESIDUAL_ONLY
//////////////////////////////////////////////////////////////////////////////////////
    //
    // Resetting the stopping condition
    //
    // Stopping condition type is one of  DEFAULT, COMBINATION, RESIDUAL_ONLY, EIGENVALUE_ONLY  
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
    //                   condition is less than subspaceTol and the eigenvector
    //                   stopping condition is less than the square root of the 
    //                   subspaceTol.
    //
    // DEFAULT         : Currently equal to COMBINATION
    
    std::string stopCondition = "RESIDUAL_ONLY";
    
    RCeigProcedure.setStopCondition(stopCondition);
   
    
    dimension             = vTmp.getDimension();   
    subspaceTol           = 1.0e-4;              // Sets the stopping tolerance for the eigenvectors 
                                                 // when RESIDUAL_ONLY is specified for stop condition            
    subspaceIncrementSize = 10;                    
    bufferSize            = 3;                                                                  
    eigCount              = dimension < 10 ? dimension : 10;

    // Remove previous results 
    
    eigValues.clear();
    eigVectors.clear();
    
    printf("\n\nXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n\n");
    
    printf("XXXX   LaplaceOp2dEigTest_SparseMatrix Results XXXX\n\n");
    printf("XXXX   Using RESIDUAL_ONLY stopping condition \n");
    printf("XXXX   Tolerance Specified : %10.5e\n\n",subspaceTol);
    printf("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n\n");
    
    RCeigProcedure.getMinEigenSystem(eigCount, subspaceTol, subspaceIncrementSize, bufferSize, vTmp,
    		sOp, randomOp, eigValues, eigVectors);

    printf("\nXXXX   Eigenvalue errors using stopping condition RESIDUAL_ONLY \n");
    printf("XXXX   Expected maximal error is (tolerance specified)^2 =  %10.5e \n\n",subspaceTol*subspaceTol);

    printf("       Eigenvalue              Error       Relative Error \n");
    
    for(long  k = 0; k < eigCount; k++ )
    {
    	printf("%-5ld %-20.14e  %10.5e   %-10.5e\n", k+1, 
    	eigValues[k], std::abs(eigValues[k] - exactEigValues[k]),std::abs(eigValues[k] -exactEigValues[k])/std::abs(exactEigValues[k]));
    }
    
    //
    // Check error in the second distinct computed eigenvector (k1 = 2 and k2 = 2)
    //
    
    for(long i = 0; i < nx; i++)
    {
    for(long j = 0; j < ny; j++)
    {
    x = xMin + (i+1)*hx;
    y = yMin + (j+1)*hy;
    k = iMap.linearIndex(i,j);
    exactEigVector(k) = std::sin((2.0*pi*(x-xMin))/(xMax-xMin))*std::sin((2.0*pi*(y-yMin))/(yMax-yMin));
    }}
    
    // normalize 
    
    exactEigVector *= 1.0/exactEigVector.norm2();
    
    // fix up the sign if necessary 

    if(exactEigVector.dot(eigVectors[3]) < 0) {exactEigVector *= -1.0;}
    
    // Compute the error 
    
    eigVecError =  exactEigVector;
    eigVecError -= eigVectors[3];
    
    eigVecErrorL2   = eigVecError.norm2();
    eigVecErrorInf = eigVecError.normInf();
    
    printf("\nXXXX   Eigenvector errors using stopping condition RESIDUAL_ONLY\n");
    printf("XXXX   Expected maximal error is tolerance specified =  %10.5e \n\n",subspaceTol);
    
    printf("Eigenvector (2,2) error (L2)  : %10.5e \n",eigVecErrorL2);
    printf("Eigenvector (2,2) error (Inf) : %10.5e \n",eigVecErrorInf);

	printf("\nXXX Execution Completed XXXX\n");
	return 0;

}


/*
 *  This routine setUpSparseOp sets up a sparse matrix representation of 
 *  the 5-point finite difference discrete Laplace operator with Dirichlet 
 *  boundary conditions for a uniform rectangular grid on a rectangular region.
 *
 *  The apply operator of the sparse matrix class applies the 
 *  difference operator
 *
 *          alpha*[(D+D-)_x + (D+D-)_y]
 *
 *  to the vector of values associated with the interior values of a grid 
 *  for the domain [xMin,xMax] X [yMin,yMax] that has xPanels panels 
 *  in the x-direction and yPanels panels in the y-direction.
 *
 *  The dimension of the linear system n = (xPanels-1)*(yPanels-1)
 *
 */

void setUpSparseOp(long xPanels, double xMin, double xMax, 
long yPanels, double yMin, double yMax, double alpha,SCC::SparseOp<SCC::DoubleVector1d>& sOp)
{
    long   nx = xPanels -1; // number of interior points in x-direction
    long   ny = yPanels -1; // number of interior points in y-direction 
    
    long   n = nx*ny; // dimension of the linear system
    
	double hx   = (xMax-xMin)/(double)(xPanels);
	double hy   = (yMax-yMin)/(double)(yPanels);

    long   rowDimension  = n;
    long   colDimension  = n;
    

    sOp.initialize(rowDimension, colDimension);

    // Instantiate a class that maps indices from the vector of values associated
    // with the discretization to the linear index associated with the
    // matrix representation and the inverse of this mapping.
    
    
    SCC::IndexMap2d iMap(nx,ny);
        
    // Construct the matrix entries. The finite difference operator being expressed 
    // is that indicated by the commments within each loop. 

	long i; long j; long k;

	// Interior grid points not adjacent to the edge

	for(i = 1; i < nx-1; i++)
	{
	for(j = 1; j < ny-1; j++)
	{
	
	//
	//V(i,j) =  alpha*((Vtmp(i+1,j) - 2.0*Vtmp(i,j) + Vtmp(i-1,j))/(hx*hx)
    //               + (Vtmp(i,j+1) - 2.0*Vtmp(i,j) + Vtmp(i,j-1))/(hy*hy));

	k = iMap.linearIndex(i,j);

    sOp(k,iMap.linearIndex(i,j)) = alpha*((-2.0)/(hx*hx) + (-2.0)/(hy*hy));

	sOp(k,iMap.linearIndex(i+1,j)) = alpha*((1.0)/(hx*hx));
	sOp(k,iMap.linearIndex(i-1,j)) = alpha*((1.0)/(hx*hx));

	sOp(k,iMap.linearIndex(i,j+1)) = alpha*((1.0)/(hy*hy));
	sOp(k,iMap.linearIndex(i,j-1)) = alpha*((1.0)/(hy*hy));
	}
	}

	// Bottom and Top

	for(i = 1; i < nx-1; i++)
	{
	j = 0;
	
	//V(i,j) =  alpha*((Vtmp(i+1,j) - 2.0*Vtmp(i,j) + Vtmp(i-1,j))/(hx*hx)
	//		         + (Vtmp(i,j+1) - 2.0*Vtmp(i,j))/(hy*hy));
	
	k = iMap.linearIndex(i,j);
    sOp(k,iMap.linearIndex(i,j)) = alpha*((-2.0)/(hx*hx) + (-2.0)/(hy*hy));
	sOp(k,iMap.linearIndex(i+1,j)) =  alpha*((1.0)/(hx*hx));
	sOp(k,iMap.linearIndex(i-1,j)) =  alpha*((1.0)/(hx*hx));
	sOp(k,iMap.linearIndex(i,j+1)) =  alpha*((1.0)/(hy*hy));


	j = ny-1;

	//V(i,j) =  alpha*((Vtmp(i+1,j) - 2.0*Vtmp(i,j) + Vtmp(i-1,j))/(hx*hx)
	//		                     + (- 2.0*Vtmp(i,j) + Vtmp(i,j-1))/(hy*hy));
	
	k = iMap.linearIndex(i,j);
    sOp(k,iMap.linearIndex(i,j))   =  alpha*((-2.0)/(hx*hx) + (-2.0)/(hy*hy));
	sOp(k,iMap.linearIndex(i+1,j)) =  alpha*((1.0)/(hx*hx));
	sOp(k,iMap.linearIndex(i-1,j)) =  alpha*((1.0)/(hx*hx));
	sOp(k,iMap.linearIndex(i,j-1)) =  alpha*((1.0)/(hy*hy));
	}

	// Left and right

	// Interior grid points not adjacent to the edge

	for(j = 1; j < ny-1; j++)
	{
	i = 0;
	
	//V(i,j) =  alpha*((Vtmp(i+1,j) - 2.0*Vtmp(i,j))/(hx*hx)
	//		         + (Vtmp(i,j+1) - 2.0*Vtmp(i,j) + Vtmp(i,j-1))/(hy*hy));
	
	k = iMap.linearIndex(i,j);
    sOp(k,iMap.linearIndex(i,j)) =    alpha*((-2.0)/(hx*hx) + (-2.0)/(hy*hy));
	sOp(k,iMap.linearIndex(i+1,j)) =  alpha*((1.0)/(hx*hx));
	sOp(k,iMap.linearIndex(i,j+1)) =  alpha*((1.0)/(hy*hy));
	sOp(k,iMap.linearIndex(i,j-1)) =  alpha*((1.0)/(hy*hy));

	i = nx-1;
	
	//V(i,j) =       alpha*((-2.0*Vtmp(i,j) + Vtmp(i-1,j))/(hx*hx)
	//		+  (Vtmp(i,j+1) - 2.0*Vtmp(i,j) + Vtmp(i,j-1))/(hy*hy));
	
	k = iMap.linearIndex(i,j);
    sOp(k,iMap.linearIndex(i,j)) = alpha*((-2.0)/(hx*hx) + (-2.0)/(hy*hy));
	sOp(k,iMap.linearIndex(i-1,j)) = alpha*((1.0)/(hx*hx));
	sOp(k,iMap.linearIndex(i,j+1)) = alpha*((1.0)/(hy*hy));
	sOp(k,iMap.linearIndex(i,j-1)) = alpha*((1.0)/(hy*hy));
	}

    // Corners

	i = 0; j = 0;
	
	//V(i,j) =  alpha*((Vtmp(i+1,j) - 2.0*Vtmp(i,j))/(hx*hx)
	//		         + (Vtmp(i,j+1) - 2.0*Vtmp(i,j))/(hy*hy));
	
	k = iMap.linearIndex(i,j);
    sOp(k,iMap.linearIndex(i,j)) =alpha*((-2.0)/(hx*hx) + (-2.0)/(hy*hy));
	sOp(k,iMap.linearIndex(i+1,j)) = alpha*((1.0)/(hx*hx));
	sOp(k,iMap.linearIndex(i,j+1)) = alpha*((1.0)/(hy*hy));

	i = 0;
	j = ny-1;
	
	//V(i,j) =  alpha*((Vtmp(i+1,j) - 2.0*Vtmp(i,j))/(hx*hx)
	//		        +  (-2.0*Vtmp(i,j) + Vtmp(i,j-1))/(hy*hy));
	
	k = iMap.linearIndex(i,j);
    sOp(k,iMap.linearIndex(i,j))  =  alpha*((-2.0)/(hx*hx) + (-2.0)/(hy*hy));
	sOp(k,iMap.linearIndex(i+1,j)) = alpha*((1.0)/(hx*hx));
	sOp(k,iMap.linearIndex(i,j-1)) = alpha*((1.0)/(hy*hy));

	i = nx-1;
	j = 0;


	//V(i,j) =  alpha*((-2.0*Vtmp(i,j) + Vtmp(i-1,j))/(hx*hx)
    //               + (Vtmp(i,j+1) - 2.0*Vtmp(i,j))/(hy*hy));
    
	k = iMap.linearIndex(i,j);
    sOp(k,iMap.linearIndex(i,j))   = alpha*((-2.0)/(hx*hx) + (-2.0)/(hy*hy));
	sOp(k,iMap.linearIndex(i-1,j)) = alpha*((1.0)/(hx*hx));
	sOp(k,iMap.linearIndex(i,j+1)) = alpha*((1.0)/(hy*hy));


	i = nx-1;
	j = ny-1;

	//V(i,j) =  alpha*((-2.0*Vtmp(i,j) + Vtmp(i-1,j))/(hx*hx)
	//    	         + (-2.0*Vtmp(i,j) + Vtmp(i,j-1))/(hy*hy));
	
	k = iMap.linearIndex(i,j);
    sOp(k,iMap.linearIndex(i,j)) =alpha*((-2.0)/(hx*hx) + (-2.0)/(hy*hy));
	sOp(k,iMap.linearIndex(i-1,j)) = alpha*((1.0)/(hx*hx));
	sOp(k,iMap.linearIndex(i,j-1)) = alpha*((1.0)/(hy*hy));

    // Compact the data and then sort the column indices in each row to
	// improve performance of the apply operator (optional)

    sOp.compact();
    sOp.sortColumnIndices();

    /*
    // Test code for matrix representation
    
    double pi     =  3.141592653589793238;

    SCC::DoubleVector1d w(n);
    SCC::DoubleVector1d wExact(n);
    SCC::DoubleVector1d wErr(n);
    
    long kx = 2;
    long ky = 3;
    
    double x; double y;

	for(long i = 0; i < nx; i++)
	{
	for(long j = 0; j < ny; j++)
	{
		x = xMin + (i+1)*hx;
		y = yMin + (j+1)*hy;
		k = iMap.linearIndex(i,j);
		w(k) = std::sin(kx*pi*(x-xMin)/LX)*std::sin(ky*pi*(y-yMin)/LY);
	}}

   	double lambda =
			alpha*(
			(2.0/(hx*hx))*(std::cos((kx*pi*hx)/LX) - 1.0)
		  + (2.0/(hy*hy))*(std::cos((ky*pi*hy)/LY) - 1.0));

	wExact = lambda*w;

	sOp.apply(w);

	// Check results(*)
	//
	// * Using a grid scaled L_2 norm so the norm is well-behaved as the number of 
	//   panels is increased.
	//

	wErr = wExact - w;

	std::cout << std::endl;
	std::cout << "XXXX  Laplacian 2D Matrix-Operator Test Output XXXX "  << std::endl;

	std::cout << "X-Panel Count : " << xPanels << std::endl;
	std::cout << "X-Wavenumber  : " << kx << std::endl;

	std::cout << "Y-Panel Count : " << yPanels << std::endl;
	std::cout << "Y-Wavenumber  : " << ky << std::endl;

	std::cout << "L_2    Error in operator = " << std::sqrt(hx*hy)*wErr.norm2()  << std::endl;
	std::cout << "L_Inf  Error in operator = " << wErr.normInf() << std::endl;
    */
}


                   
                   
                   
                   

