#include <iostream>
#include <cmath>

#include "GridFunctionNd/SCC_GridFunction2d.h"

/**
 *                       Class LaplaceOp2d
 *  A class whose apply member function implements the 5-point finite difference
 *  discrete Laplace operator with Dirichlet boundary conditions.
 *
 *  The apply operator of this class applies the difference operator
 *
 *          alpha*[(D+D-)_x + (D+D-)_y]
 *
 *  at all interior values of an SCC::GridFunction2d and sets the boundary values to 0.0
 */

#ifndef LAPLACE_OP_2D_
#define LAPLACE_OP_2D_

class LaplaceOp2d
{
public:

	LaplaceOp2d()
	{
	this->alpha = 0.0;
	}

	virtual ~LaplaceOp2d()
	{}
	
	LaplaceOp2d(const LaplaceOp2d& L)
	{
		initialize(L);
	}

	LaplaceOp2d(double alpha)
	{
		this->alpha   = alpha;
	}
	
	void initialize()
	{
		this->alpha = 0;
	}
	
	void initialize(const LaplaceOp2d& L)
	{
		this->alpha = L.alpha;
	}

	void initialize(double alpha)
	{
    this->alpha   = alpha;
	}
	/**
	This routine applies alpha times the 5-point discrete Laplace operator to
	the interior grid points associated with a uniform discretization
	of a 2D rectangular domain.

	Input  : SCC_GriddFunction2d class instance whose values are those of a uniform 2D grid.
	         The function values associated with this class are the values at both the interior
	         and boundary points of the discretization.

    Output : The interior values of the input GridFunction are overwritten
             with the 5-point difference approximation boundary values are
             set to 0.

    If _DEBUG is defined at compile time, bounds checking is performed.
	*/

	void apply(SCC::GridFunction2d& V)
	{

	// Capture values since we are over-writing the input vector
    //
	// Note: The use of initialize instead of = here is because the initialize
	// member function is "smart" in the sense that if the existing instance
	// of Vtmp has identical size to the input V, then it just copies
	// over the values, and doesn't destroy and recreate a new instance
	// of the GridFunction2d.
	//

	Vtmp.initialize(V);

    // Extract grid information from V

    double hx = V.getHx();
    double hy = V.getHy();
    long  xPanels = V.getXpanelCount();
    long  yPanels = V.getYpanelCount();

	long i; long j;

	// Interior grid points not adjacent to the edge

	for(i = 1; i < xPanels; i++)
	{
	for(j = 1; j < yPanels; j++)
	{
	V(i,j) =  alpha*((Vtmp(i+1,j) - 2.0*Vtmp(i,j) + Vtmp(i-1,j))/(hx*hx)
                   + (Vtmp(i,j+1) - 2.0*Vtmp(i,j) + Vtmp(i,j-1))/(hy*hy));
	}
	}

	V.setBoundaryValues(0.0);

	}

    double alpha;             // Coefficient of the discrete Laplace operator
	SCC::GridFunction2d Vtmp; // Temporary
};

#endif



