/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _POLYGON_SET_H
#define _POLYGON_SET_H
/****************************************************************************\ 
 **                                                            (C)2000 RUS   **
 **                                                                          **
 ** Description: Simple Example how to create a set of polygons              **
 **                                                                          **
 ** Name:        PolygonSet                                                  **
 ** Category:    examples                                                    **
 **                                                                          **
 ** Author: D. Rainer		                                                **
 **                                                                          **
 ** History:  								                                **
 ** April-00     					       		                            **
 **                                                                          **
 **                                                                          **
\****************************************************************************/

#include <api/coModule.h>
using namespace covise;
namespace covise
{
class coDoPolygons;
}

class PolygonSet : public coModule
{

private:
    // compute callback
    virtual int compute(const char *port);

    coDoPolygons *createCube(char *objectName, float ox, float oy, float oz, float size);

    coIntSliderParam *p_numObjects;
    coBooleanParam *p_timeSteps;

    coOutputPort *outPort_polySet;

public:
    PolygonSet(int argc, char *argv[]);
    virtual ~PolygonSet();
};
#endif
