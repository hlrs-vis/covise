/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// xx.yy.2002 / 1 / Carbo.h

#ifndef _CARBO_H
#define _CARBO_H

//#include <math.h>
//#include <iostream.h>
//#include <stdlib.h>
//#include <stdio.h>
//#include <ctype.h>
//#include <vector.h>
//#include <string.h>

#include "nrutil.h"

class Carbo
{
    Carbo();
    ~Carbo();
};

void carbonDioxide(fvec &ePotential, f2ten &eField, float ucharge, float size, const f2ten &coord);
#endif
