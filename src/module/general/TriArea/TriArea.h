/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _TRIAREA_H
#define _TRIAREA_H

#include <api/coSimpleModule.h>
using namespace covise;

#include <util/coviseCompat.h>

// Calculate triangle area or ratio of edge lengths
class TriArea : public coSimpleModule
{
    static const char *s_defMapNames[];
    coChoiceParam *p_areamode;
    enum AreaSelectMap
    {
        TriangleArea = 0,
        RatioMinToMaxSide = 1,
        AngleNormalToCamera = 2,
        SomeThingElse = 3
    } areamode;

private:
    virtual int compute(const char *port);

    ////////// ports
    coInputPort *p_inPort;
    coOutputPort *p_outPort;
    coFloatVectorParam *p_cameraPosition; 

    //  Shared memory data
    coDistributedObject *tri_area(const coDistributedObject *m, float *cameraPosition, const char *objName);

public:
    TriArea(int argc, char *argv[]);
};
#endif // _TRIAREA_H
