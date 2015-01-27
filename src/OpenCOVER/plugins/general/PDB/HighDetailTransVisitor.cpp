/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <iostream>
#include "HighDetailTransVisitor.h"

HighDetailTransVisitor::HighDetailTransVisitor()
    : TransparentVisitor()
{
    _highdetail = true;
    _distance = 1;
    _area = 1;
}

// High detail code
void HighDetailTransVisitor::calculateAlphaAndBin(float dst2view)
{

    float LOW = _distance * 0.98;
    float HIGH = LOW + _area;

    if (dst2view < LOW)
    {
        _transparent = false;
        _alpha = 0;
    }
    else if (dst2view >= HIGH)
    {
        _transparent = true;
        _alpha = 1;
    }
    // in between
    else
    {
        _transparent = true;
        _alpha = (dst2view - LOW) / (HIGH - LOW);

        if (_alpha > 0.85)
            _alpha = 1;
    }
}
