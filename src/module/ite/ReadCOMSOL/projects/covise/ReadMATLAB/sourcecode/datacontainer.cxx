/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// ITE-Toolbox ReadMATLAB (C) Institute for Theory of Electrical Engineering
//
// Data container

#include "datacontainer.hxx"
#include <stdlib.h>

DataContainer::DataContainer(const unsigned long noPoints)
    : _noPoints(noPoints)
    , _isSet(false)
    , _isVector(false)
    , _x(NULL)
    , _y(NULL)
    , _z(NULL)
{
}

DataContainer::~DataContainer()
{
    if (_x != NULL)
        delete[] _x;
    if (_y != NULL)
        delete[] _y;
    if (_z != NULL)
        delete[] _z;
}

bool DataContainer::SetType(const bool isVector)
{
    bool retVal = true;
    if (_isSet)
        retVal = false;
    else
    {
        _isSet = true;
        _isVector = isVector;
        _x = new double[_noPoints];
        if (_isVector)
        {
            _y = new double[_noPoints];
            _z = new double[_noPoints];
        }
    }
    return retVal;
}

bool DataContainer::IsVector(void) const
{
    return _isVector;
}

bool DataContainer::IsSet(void) const
{
    return _isSet;
}

double *DataContainer::GetX(void)
{
    return _x;
}

double *DataContainer::GetY(void)
{
    return _y;
}

double *DataContainer::GetZ(void)
{
    return _z;
}
