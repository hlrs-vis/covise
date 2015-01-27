/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#if !defined(__Animate_H)
#define __Animate_H

// includes
#include <api/coModule.h>
using namespace covise;
#include <do/coDoSet.h>
#include <do/coDoData.h>
#include <do/coDoUnstructuredGrid.h>
#include <do/coDoPolygons.h>

class Animate : public coModule
{

private:
    virtual int compute(const char *port);

    int Diagnose();
    coInputPort *p_inGrid;
    coInputPort *p_inVelo;
    coOutputPort *p_outGrid;
    coOutputPort *p_outVelo;
    coFloatParam *p_angle;
    coFloatParam *p_startAngle;
    coFloatVectorParam *p_axis;
    coFloatVectorParam *p_pos;
    float currentAngle;
    int numTimesteps;
    bool polygons;
    bool trianglestrips;
    coDoVec3 *veloInObj;
    coDoVec3 *veloOutObj;
    coDoUnstructuredGrid *gridInObj;
    coDoUnstructuredGrid *gridOutObj;
    coDoPolygons *polyInObj;
    coDoPolygons *polyOutObj;
    coDoSet *setGeoIn;
    coDoSet *setIn;
    coDoSet *setOut;
    coDoSet *setVeloOut;

public:
    Animate(int argc, char *argv[]);
};
#endif // __Animate_H
