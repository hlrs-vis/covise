/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _CU_CUTTING_SURFACE_USG_H
#define _CU_CUTTING_SURFACE_USG_H

#include <api/coModule.h>
class State;

struct polyData
{

    covise::coDistributedObject *polygons;
    covise::coDistributedObject *data;
};

struct ltstr
{
    bool operator()(const char *s1, const char *s2) const
    {
        return strcmp(s1, s2) < 0;
    }
};

class cuCuttingSurfaceUSG : public covise::coModule
{

public:
    cuCuttingSurfaceUSG(int argc, char **argv);
    virtual ~cuCuttingSurfaceUSG();

private:
    virtual int compute(const char *port);
    virtual void param(const char *name, bool inMapLoading);

    struct polyData createPolygons(covise::coOutputPort *polyPort,
                                   covise::coOutputPort *dataPort,
                                   covise::coDistributedObject *obj,
                                   covise::coDistributedObject *data_obj,
                                   float nx, float ny, float nz, float dist,
                                   int level = 0);

    covise::coInputPort *p_mesh;
    covise::coOutputPort *p_polygons;
    covise::coInputPort *p_dataIn;
    covise::coInputPort *p_mapIn;
    covise::coOutputPort *p_data;
    covise::coFloatSliderParam *p_threshold;

    covise::coFloatVectorParam *p_vertex;
    covise::coFloatParam *p_scalar;

    std::string objectName;
    State *state;
};

#endif //_CU_CUTTING_SURFACE_USG_H
