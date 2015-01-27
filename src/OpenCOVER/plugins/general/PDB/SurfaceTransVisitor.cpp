/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <iostream>
#include "SurfaceTransVisitor.h"

SurfaceTransVisitor::SurfaceTransVisitor()
    : TransparentVisitor()
{
    _highdetail = false;
    _distance = 1;
    _area = 1;
}

// low detail code
void LowDetailTransVisitor::calculateAlphaAndBin(float dst2view)
{

    float LOW = _distance * 1.02;
    float HIGH = LOW + _area;
    // enable opacity and display high detail structure
    if (dst2view < LOW)
    {
        _transparent = true;
        _alpha = 1;
    }
    // enable opacity and display low detail structure
    else if (dst2view >= HIGH)
    {
        _transparent = false;
        _alpha = 0;
    }
    // in between
    else
    {
        _transparent = true;
        _alpha = 1 - ((dst2view - LOW) / (HIGH - LOW));
        if (_alpha > 0.85)
            _alpha = 1;
    }
}
