/*
 * AvectorClass.h
 *
 * A wrapper class for std::vector<double> that adds the minimal number of
 * vector operations required to instantiate instances of the
 * RayleighChebyshev class.
 *
 *
 * Note : Individual vector element access is not needed by the RayleighChebyshev
 * procedure; the eigensystem procedure implemented in the RayleighChebyshev
 * procedure is therefore agnostic to any particular representation of the data
 * within the vector.
 *
 *
 *  Created on: Oct 11, 2024
 *      Author: anderson
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

#include <vector>
#include <cmath>

#ifndef AVECTORCLASS_
#define AVECTORCLASS_

class AvectorClass
{
	public:

//////////////////////////////////////////////////////////
//  Required constructors
//////////////////////////////////////////////////////////

	AvectorClass()
	{
	    vData.clear();
	}

    AvectorClass(const AvectorClass& W)
    {
        initialize(W);
    }

//////////////////////////////////////////////////////////
// Constructor not required for RayleighChebyshve
// but useful for creating test program
//////////////////////////////////////////////////////////

    AvectorClass(long dimension)
    {
        vData.resize(dimension);
    }


	virtual ~AvectorClass(){}

//////////////////////////////////////////////////////////////////
//  Member functions required to use this class as a
//  RayleighChebyshev template parameter
//////////////////////////////////////////////////////////////////

	void initialize(const AvectorClass& W)
	{
	    vData = W.vData;
	}

	double dot(const AvectorClass& W) const
	{
		double dotSum = 0.0;
		for(size_t k = 0; k < vData.size(); k++)
		{
		dotSum += vData[k]*W.vData[k];
		}
		return dotSum;
	}

	void operator *=(double alpha)
	{
	    for(size_t k = 0; k < vData.size(); k++)
	    {
	       vData[k] *= alpha;
	    }
	}

    void operator +=(const AvectorClass& W)
	{
	    for(size_t k = 0; k < vData.size(); k++)
	    {
	        vData[k] += W.vData[k];
	    }
	}

	void operator -=(const AvectorClass& W)
	{
	    for(size_t k = 0; k < vData.size(); k++)
	    {
	        vData[k] -= W.vData[k];
	    }
	}

	double norm2() const
	{
		double normSquared =  (*this).dot(*this);
		return std::sqrt(std::abs(normSquared));
	}

	size_t getDimension() const
	{
	return vData.size();
	}

	std::vector<double> vData;
};



#endif /* AVECTORCLASS_ */
