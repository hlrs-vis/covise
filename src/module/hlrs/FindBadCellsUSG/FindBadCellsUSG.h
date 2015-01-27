/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _FIND_BAD_CELLS_USG_H
#define _FIND_BAD_CELLS_USG_H

#include <api/coModule.h>
#include <do/coDoUnstructuredGrid.h>
#include <do/coDoSet.h>
#include <do/coDoData.h>

using namespace covise;

struct polyData
{

    coDistributedObject *polygons;
    coDistributedObject *data;
};

class FindBadCellsUSG : public coModule
{

public:
    FindBadCellsUSG(int argc, char **argv);
    virtual ~FindBadCellsUSG();

private:
    virtual int compute(const char *port);
    virtual void param(const char *name, bool inMapLoading);

    struct polyData createPolygons(coOutputPort *polyPort,
                                   coOutputPort *dataPort,
                                   const coDistributedObject *obj,
                                   float threshold,
                                   int level = 0);

    float testTetrahedron(int elemIndex, int *elemList, int *connList, float *x, float *y, float *z, float threshold);
    float testPrism(int elemIndex, int *elemList, int *connList, float *x, float *y, float *z, float threshold);
    float testHexaeder(int elemIndex, int *elemList, int *connList, float *x, float *y, float *z, float threshold);

    coInputPort *p_mesh;
    coOutputPort *p_polygons;
    coOutputPort *p_data;
    coFloatSliderParam *p_threshold;
};

#endif
