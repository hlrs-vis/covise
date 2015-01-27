/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <iostream>
#include "LowDetailTransVisitor.h"

LowDetailTransVisitor::LowDetailTransVisitor()
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

    //cerr << "Distance " << dst2view << ": Low range " << LOW << ": High range " << HIGH << endl;
    // enable opacity and display high detail structure
    if (dst2view < LOW)
    {
        //cerr << "Less than low" << endl;
        _transparent = true;
        _alpha = 1;
    }
    // enable opacity and display low detail structure
    else if (dst2view >= HIGH)
    {
        //cerr << "Higher than high" << endl;
        _transparent = false;
        _alpha = 0;
    }
    // in between
    else
    {
        _transparent = true;
        _alpha = 1 - ((dst2view - LOW) / (HIGH - LOW));
        //cerr << "Alpha in between " << _alpha << endl;
    }
}
