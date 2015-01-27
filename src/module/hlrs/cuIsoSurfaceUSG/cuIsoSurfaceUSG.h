/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _CU_ISO_SURFACE_USG_H
#define _CU_ISO_SURFACE_USG_H

#include <api/coModule.h>
class State;

struct polyData
{

    covise::coDistributedObject *polygons;
    covise::coDistributedObject *data;
};

class cuIsoSurfaceUSG : public covise::coModule
{

public:
    cuIsoSurfaceUSG(int argc, char **argv);
    virtual ~cuIsoSurfaceUSG();

private:
    virtual int compute(const char *port);
    virtual void param(const char *name, bool inMapLoading);

    struct polyData createPolygons(covise::coOutputPort *polyPort,
                                   covise::coOutputPort *dataPort,
                                   covise::coDistributedObject *obj,
                                   covise::coDistributedObject *data_obj,
                                   float threshold,
                                   int level = 0);

    float testTetrahedron(int elemIndex, int *elemList, int *connList, float *x, float *y, float *z, float threshold);

    float testPrism(int elemIndex, int *elemList, int *connList, float *x, float *y, float *z, float threshold);

    covise::coInputPort *p_mesh;
    covise::coOutputPort *p_polygons;
    covise::coInputPort *p_dataIn;
    covise::coInputPort *p_mapIn;
    covise::coOutputPort *p_data;
    covise::coFloatSliderParam *p_threshold;

    std::string objectName;
    State *state;
};

#endif //_CU_ISO_SURFACE_USG_H
