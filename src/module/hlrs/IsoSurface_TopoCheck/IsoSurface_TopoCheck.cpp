/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\ 
 **                                                                          **
 ** Description: IsoSurface for UniformGrids                                 **
 **              uses Thomas Lewiner's algorithm that checks                 **
 **              for topologically correct results                           **
 **                                                                          **
 ** Name:        IsoSurface_TopoCheck                                        **
 ** Category:    Mapper                                                      **
 **                                                                          **
 ** Author: M. Becker                                                        **
 ** 01/2009                                                                  **
 **                                                                          **
 **                                                                          **
\****************************************************************************/

#include "IsoSurface_TopoCheck.h"
#include <stdlib.h>
#include <stdio.h>
#include <do/coDoPolygons.h>
#include <do/coDoUniformGrid.h>
#include <do/coDoData.h>

IsoSurface_TopoCheck::IsoSurface_TopoCheck(int argc, char *argv[])
    : coModule(argc, argv, "IsoSurface module for uniform structured grids, generates topologically checked isosurfaces")
{

    // input ports
    p_gridIn = addInputPort("gridIn", "UniformGrid", "input grid");
    p_dataIn = addInputPort("dataIn", "Float", "input data");

    // output port
    p_polyOut = addOutputPort("polygonsOut", "Polygons", "polygons which form the cubes");
    p_normalsOut = addOutputPort("normals", "Vec3", "normal vectors");

    // input parameters
    // parameters:

    p_isovalue = addFloatParam("isovalue", "the isovalue");
    p_isovalue->setValue(0.);
}

void IsoSurface_TopoCheck::postInst()
{
}

int IsoSurface_TopoCheck::compute(const char * /*port*/)
{
    printf("entering compute ...\n");

    const coDoUniformGrid *gridIn = NULL;
    gridIn = dynamic_cast<const coDoUniformGrid *>(p_gridIn->getCurrentObject());

    const coDoFloat *dataIn = NULL;
    dataIn = dynamic_cast<const coDoFloat *>(p_dataIn->getCurrentObject());

    int nx, ny, nz;
    float dx, dy, dz;
    float x_min, x_max, y_min, y_max, z_min, z_max;
    gridIn->getGridSize(&nx, &ny, &nz);
    gridIn->getDelta(&dx, &dy, &dz);
    gridIn->getMinMax(&x_min, &x_max, &y_min, &y_max, &z_min, &z_max);

    sendInfo("grid size is x:%d y:%d z:%d", nx, ny, nz);
    sendInfo("grid delta is dx:%f dy:%f dz:%f", dx, dy, dz);
    sendInfo("xmin=%f, xmax=%f, ymin=%f, ymax=%f, zmin=%f, zmax=%f", x_min, x_max, y_min, y_max, z_min, z_max);

    float *data;
    dataIn->getAddress(&data);

    int nverts, ntrigs;
    MarchingCubes mc;

    float *xcoord, *ycoord, *zcoord;
    float *normx, *normy, *normz;
    int *vl, *pl;

    mc.lewiner(p_isovalue->getValue(), nx, ny, nz,
               x_min, y_min, z_min,
               dx, dy, dz,
               data,
               nverts,
               ntrigs,
               xcoord, ycoord, zcoord,
               normx, normy, normz,
               vl, pl);

    coDoPolygons *polyOut = new coDoPolygons(p_polyOut->getObjName(), nverts, xcoord, ycoord, zcoord,
                                             ntrigs * 3, vl, ntrigs, pl);

    coDoVec3 *normals = new coDoVec3(p_normalsOut->getObjName(), nverts, normx, normy, normz);

    p_polyOut->setCurrentObject(polyOut);
    p_normalsOut->setCurrentObject(normals);

    return SUCCESS;
}

IsoSurface_TopoCheck::~IsoSurface_TopoCheck()
{
}

void IsoSurface_TopoCheck::param(const char *name, bool inMapLoading)
{
    (void)name;
    (void)inMapLoading;
}

MODULE_MAIN(Mapper, IsoSurface_TopoCheck)
