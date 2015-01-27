/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\ 
 **                                                            (C)2009 HLRS  **
 **                                                                          **
 ** Description: Use scalar data array to displace polygon mesh              **
 **              in normal direction                                         **
 **                                                                          **
 ** Name:        DataHeight                                                  **                                                        **
 ** Category:    Mapper                                                      **
 **                                                                          **
 ** Author: M. Becker                                                        **
 **                                                                          **
 **                                                                          **
 **                                                                          **
\****************************************************************************/

#include "DataHeight.h"
//#include <stdlib.h>
//#include <stdio.h>
//#include <string.h>
#include <do/coDoPolygons.h>
#include <do/coDoData.h>
#include <alg/coFeatureLines.h>

DataHeight::DataHeight(int argc, char *argv[])
    : coModule(argc, argv, "display scalar data as height map")
{

    // input ports
    p_polyIn = addInputPort("poly", "Polygons", "input polygons");
    p_dataIn = addInputPort("data", "Float", "input data");
    p_normalsIn = addInputPort("normals", "Vec3", "input data");

    // output port
    p_polyOut = addOutputPort("polyIn", "Polygons", "polygons displaced due to data");

    // parameters:
    p_scale = addFloatParam("scale", "data scale");
    p_scale->setValue(1.);
}

void DataHeight::postInst()
{
}

int DataHeight::compute(const char * /*port*/)
{

    coDoVec3 *normalsIn = dynamic_cast<coDoVec3 *>(p_normalsIn->getCurrentObject());
    float *nx = NULL, *ny = NULL, *nz = NULL;
    normalsIn->getAddresses(&nx, &ny, &nz);

    coDistributedObject *GeoObj = p_polyIn->getCurrentObject();
    coDoPolygons *polyIn = dynamic_cast<coDoPolygons *>(GeoObj);

    if (!GeoObj->isType("POLYGN"))
    {
        sendError("module only works with polygons as input geometry.");
        return STOP_PIPELINE;
    }

    float *data = NULL;
    coDoFloat *dataIn = dynamic_cast<coDoFloat *>(p_dataIn->getCurrentObject());
    dataIn->getAddress(&data);

    float *x = NULL, *y = NULL, *z = NULL;
    int *vl = NULL, *pl = NULL;
    polyIn->getAddresses(&x, &y, &z, &vl, &pl);

    std::vector<int> tri_vl;
    std::vector<int> tri_pl;
    std::vector<int> tri_codes;

    coFeatureLines *myFeature;

    myFeature->Triangulate(tri_vl, // tri_vl is the output.
                           tri_codes, // for each triangle we get the label of the polygon it is part of.
                           polyIn->getNumPolygons(), // number of original polygons.
                           polyIn->getNumVertices(), // length of the connectivity list.
                           pl,
                           vl,
                           x, y, z);

    int numTrias = tri_vl.size() / 3;
    int numPoints = polyIn->getNumPoints();

    float *x_out = new float[numPoints];
    float *y_out = new float[numPoints];
    float *z_out = new float[numPoints];
    memcpy(x_out, x, numPoints * sizeof(float));
    memcpy(y_out, y, numPoints * sizeof(float));
    memcpy(z_out, z, numPoints * sizeof(float));

    for (int i = 0; i < numTrias; i++)
    {
        tri_pl.push_back(3 * i);
    }

    float scale = p_scale->getValue();
    for (int i = 0; i < numPoints; i++)
    {
        x_out[i] += nx[i] * scale * data[i];
        y_out[i] += ny[i] * scale * data[i];
        z_out[i] += nz[i] * scale * data[i];
    }

    coDoPolygons *polyOut = new coDoPolygons(p_polyOut->getObjName(),
                                             numPoints,
                                             x_out, y_out, z_out,
                                             3 * numTrias, &tri_vl[0],
                                             numTrias, &tri_pl[0]);

    p_polyOut->setCurrentObject(polyOut);

    return SUCCESS;
}

DataHeight::~DataHeight()
{
}

void DataHeight::param(const char *name, bool /*inMapLoading*/)
{
    if (strcmp(name, p_scale->getName()) == 0)
    {
    }
}

MODULE_MAIN(Mapper, DataHeight)
