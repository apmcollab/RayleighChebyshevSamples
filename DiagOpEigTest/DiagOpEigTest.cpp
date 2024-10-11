
//XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
//                          DiagOpTest.cpp
//XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
// A test code that demonstrates the use of the RayleighChebyshev procedure
// to determine a specified number of eigenvalues and eigenvectors of
// a diagonal linear operator.
//
// Note: The convergence behavior of the RayleighChebyshev procedure is determined
// by the distribution of the eigenvalues of the linear operator. The convergence
// is not influenced by the structure of the matrix representation of the operator 
// so testing the procedure using a diagonal operator is just as useful as 
// testing the procedure on a non-diagonal matrix. 
//
// The RayleighChebyshev class is templated with respect to three classes  
//
// (a) A vector class
//
// (b) A linear operator class that acts upon instances the of the vector class
//     and whose eigensystem is to be determined.
//
// (c) A randomize operator class that when applied to an instance of the vector class
//     sets the vector elements` to have random entries.
//
//
// This sample includes and uses implementations of these classes created with the minimal 
// number of member functions that are required when the class is used as template argument
// for the RayleighChebyshev procedure. 
//
// These class are defined and implemented in the following header files
//
// AvectorClass.h        : A wrapper class for std::vector<double> that adds required member functions

// ArandomizeOpClass.h   : A class that uses the C++ random number generator to initialize
//                         AvectorClass instances with random entries

// DiagOpClass.h         : A diagonal operator
//
// These are not general purpose, but sample classes that illustrate the functionality
// required of the vector and operator classes.
//
/////////////////////////////////////////////////////////////////////////////////////////////
//
// Note: The compilation of this program does not require a Lapack installation. 
//
// Directory set up for this test program
//
//             [Samples Directory]
// -------------------------------------------------
//         |                             |                
//[DigOpEigTest]                   [Components]
//                                       |
//                               -------------------
//                              Component directories
//
// where the component directories contain the source directories:
//
// RayleighChebyshev (https://github.com/apmcollab/RayleighChebyshev)
//
// 
// The command line compilation command is
//
// g++ DiagOpEigTest.cpp -std=c++17 -D RC_WITHOUT_LAPACK_ -I../Components  -o DiagOpEigTest.exe
//
// To enable OpenMP add -fopenmp to the above command line (or the equivalent flag for your compiler).
//
// To build and run the test program using CMake, cd to the DiagOpEigTest directory and then execute
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
//  Volume 229 Issue 19, September, 2010.
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

#include "AvectorClass.h"                        // Vector class
#include "DiagOpClass.h"                         // Linear operator classs whose eigensystem is being computed
#include "ArandomizeOpClass.h"                   // Class for initializing vector with random entries

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
//                         Problem set up  
//////////////////////////////////////////////////////////////////////////////////////

    long systemSize = 200;
   
//////////////////////////////////////////////////////////////////////////////////////
//                        Operator set up.
//////////////////////////////////////////////////////////////////////////////////////

    // Instantiate linear operator class
    
    DiagOpClass    diagOp;

    // Instantiate randomize operator class

    ArandomizeOpClass randomOp;

    // Declare an instance of the Raylegh-Chebyshev eigensystem procedure

    RayleighChebyshev < AvectorClass,   DiagOpClass , ArandomizeOpClass> RCeigProcedure;
    //                         |              |                   |
    //                         |              |                   |
    //               vector class     linear operator class   randomize operator class


    // Set/unset verbose output

    RCeigProcedure.setEigDiagnosticsFlag(true);
    RCeigProcedure.setVerboseFlag(true);

//////////////////////////////////////////////////////////////////////////////////////
//                       Input/Output set up
//////////////////////////////////////////////////////////////////////////////////////

    // Creating input parameters

    AvectorClass vTmp(systemSize);                      // A required temporary vector for input. This vector must
                                                        // be a non-null instance of the vector class.

    long   dimension           = vTmp.getDimension();   // The dimension of the vector space.

    double subspaceTol         = 1.0e-6;                // The stopping tolerance.
    
    long subspaceIncrementSize = 10;                    // Subspace size used to determine the eigenpairs.
    
    long bufferSize            = 3;                     // The number of additional eigenpairs that are internally computed but not output.
                                                        //
                                                        // The size (or dimension) of the buffer subspace must be sufficiently large  
                                                        // so there is at least one eigenpair in that subspace whose eigenvalue is distinct from 
                                                        // the set of eigenvalues in the subspace containing the desired eigenpairs.

    //                                                  
    // Specifying the number of eigenpairs to compute. Typically one chooses subspaceIncrementSize to 
    // be equal to the desired number of eigenpairs. If subspaceIncrementSize is less than eigCount, then 
    // subspaceIncrementSize eigenpairs are computed incrementally until the total number of eigenpairs 
    // are computed. 


    long eigCount              = dimension < 10 ? dimension : 10;
    
    // Allocate vectors for output of eigenvectors and eigenvalues

    std::vector <AvectorClass>      eigVectors;
    std::vector <double>             eigValues;

    
//////////////////////////////////////////////////////////////////////////////////////
//           Computation of the eigensystem and evaluation of the error
//////////////////////////////////////////////////////////////////////////////////////
    
    printf("\n\nXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n\n");
    
    printf("XXXX   DiagOpEig_Test Results XXXX\n\n");
    printf("XXXX   Using default parameters\n");
    printf("XXXX   Tolerance specified : %10.5e\n\n",subspaceTol);
    
    printf("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n\n");
    


    RCeigProcedure.getMinEigenSystem(eigCount, subspaceTol, subspaceIncrementSize, bufferSize, vTmp,
    		           diagOp, randomOp, eigValues, eigVectors);
    		       
    		       
    // Exact eigenvectors of diagOp
    
    std::vector<double> exactEigValues(systemSize);
    
    for(size_t i = 0; i < exactEigValues.size(); i++)
    {
    exactEigValues[i] = (double)(i+1);
    }
    
    printf("\nXXXX   Eigenvalue errors using stopping condition DEFAULT = COMBINATION \n");
    printf("XXXX   Expected maximal error is tolerance specified =  %10.5e \n\n",subspaceTol);

    printf("       Eigenvalue              Error       Relative Error \n");
    
    for(long  k = 0; k < eigCount; k++ )
    {
        printf("%-5ld %-20.14e  %10.5e   %-10.5e\n", k+1, 
     	eigValues[k], std::abs(eigValues[k] - exactEigValues[k]),std::abs(eigValues[k] -exactEigValues[k])/std::abs(exactEigValues[k]));
    }
    
    
    // Check the eigenvector error, possible since all the eigenvectors are distinct. The 
    // eigenvectors are just the unit vectors so the error in the kth vector is just
    // the norm of the kth eigenvector with it's kth entry set to zero.
    
    printf("\nXXXX   Eigenvector errors using stopping condition DEFAULT = COMBINATION \n");
    printf("XXXX   Expected maximal error is sqrt(tolerance specified) =  %10.5e \n\n",std::sqrt(subspaceTol));
 
    
    double maxError;
    
    for(long k = 0; k < eigCount; k++)
    {

        eigVectors[k].vData[k] = 0.0;
    
        maxError = 0.0;
        for(size_t j = 0; j < eigVectors[k].getDimension(); j++)
        {
        maxError = std::max(maxError, std::abs(eigVectors[k].vData[j]));
        }
        
        printf("Eigenvector %-5ld error (Inf) :  %10.5e \n",k,maxError); 
    }
    
    
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
    subspaceTol           = 1.0e-6;
    subspaceIncrementSize = 10;                    
    bufferSize            = 3;                                                                  
    eigCount              = dimension < 10 ? dimension : 10;

    // Remove previous results 
    
    eigValues.clear();
    eigVectors.clear();
    
    printf("\n\nXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n\n");
    
    printf("XXXX   DiagOpEig_Test Results XXXX\n\n");
    printf("XXXX   Using RESIDUAL_ONLY stopping condition \n");
    printf("XXXX   Tolerance Specified : %10.5e\n\n",subspaceTol);
    
    printf("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n\n");
    
    RCeigProcedure.getMinEigenSystem(eigCount, subspaceTol, subspaceIncrementSize, bufferSize, vTmp,
    		diagOp, randomOp, eigValues, eigVectors);

    printf("\nXXXX   Eigenvalue errors using stopping condition RESIDUAL_ONLY \n");
    printf("XXXX   Expected maximal error is (tolerance specified)^2 =  %10.5e \n\n",subspaceTol*subspaceTol);
    
    printf("       Eigenvalue              Error       Relative Error \n");
    
    for(long  k = 0; k < eigCount; k++ )
    {
    	printf("%-5ld %-20.14e  %10.5e   %-10.5e\n", k+1, 
    	eigValues[k], std::abs(eigValues[k] - exactEigValues[k]),std::abs(eigValues[k] -exactEigValues[k])/std::abs(exactEigValues[k]));
    }
        
    // Check the eigenvector error, possible since all the eigenvectors are distinct. The 
    // eigenvectors are just the unit vectors so the error in the kth vector is just
    // the norm of the kth eigenvector with it's kth entry set to zero.
    
    
    printf("\nXXXX   Eigenvector errors using stopping condition RESIDUAL_ONLY\n");
    printf("XXXX   Expected maximal error is tolerance specified =  %10.5e \n\n",subspaceTol);
    
    for(long k = 0; k < eigCount; k++)
    {

        eigVectors[k].vData[k] = 0.0;
    
        maxError = 0.0;
        for(size_t j = 0; j < eigVectors[k].getDimension(); j++)
        {
        maxError = std::max(maxError, std::abs(eigVectors[k].vData[j]));
        }
        
        printf("Eigenvector %-5ld error (Inf) :  %10.5e \n",k,maxError); 
    }
    
	printf("\nXXX Execution Completed XXXX\n");
	return 0;

}

                   
                   
                   

