/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _CUT_GEOMETRY_H
#define _CUT_GEOMETRY_H

#include <stdlib.h>
#include <stdio.h>

#include <api/coSimpleModule.h>
using namespace covise;

////// our class
class LinesToPoints : public coSimpleModule
{
private:
    /////////ports
    coInputPort *p_geo_in, *p_data_in;
    coOutputPort *p_geo_out, *p_data_out;

    virtual int compute();

public:
    LinesToPoints();
};
#endif
