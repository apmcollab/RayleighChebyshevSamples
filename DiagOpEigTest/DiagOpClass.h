/*
 * DiagOpClass.h
 *
 *  Created on: Oct 11, 2024
 *      Author: anderson
 *
 *
 * This class implements an apply(...)  operator in which each entry of the input vector
 * is multiplied by it's index + 1 (indexing starting from 0).
 *
 *
 *
 */
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
#include "AvectorClass.h"

#ifndef DIAGOPCLASS_
#define DIAGOPCLASS_

class DiagOpClass
{
    public:

	DiagOpClass(){}

	// Required member function for RayleighChebyshev

	void apply(AvectorClass& V)
	{
		for(size_t i = 0; i < V.getDimension(); i++)
		{
		V.vData[i] *= (double)(i+1);
		}
	}

	// Required member function if running RayleighChebyshev with OpenMP
	//
	// This member function creates a duplicate instance of the operator.
	//
	// Since the DiagOpClass as implemented here has no data members, it is just
	// a trivial implementation here.
    //
	void initialize(DiagOpClass& D)
	{}
};


#endif /* DIAGOPCLASS_ */
