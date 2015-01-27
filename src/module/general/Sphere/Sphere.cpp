/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <alg/coSphere.h>
#include <util/coviseCompat.h>
#include "Sphere.h"
#include <do/coDoPoints.h>
#include <do/coDoTriangleStrips.h>
#include <do/coDoData.h>
#include <do/coDoStructuredGrid.h>
#include <do/coDoUnstructuredGrid.h>

Sphere::Sphere(int argc, char **argv)
    : coSimpleModule(argc, argv, "Make Spheres from points")
{
    p_pointsIn = addInputPort("points", "Points|UnstructuredGrid|StructuredGrid", "the points to be transformed into spheres");

    p_colorsIn = addInputPort("colors_in", "Float|RGBA|Vec3|IntArr|Int|Byte|Mat3|Tensor|Vec2", "data to be mapped onto the spheres");
    p_colorsIn->setRequired(0);

    p_dataIn = addInputPort("data", "Float|Vec3", "data to be added to parameter radius");
    p_dataIn->setRequired(0);

    p_spheresOut = addOutputPort("spheres", "TriangleStrips|Polygons|Spheres", "the spheres");
    p_dataOut = addOutputPort("dataOut", "Float|RGBA|Vec3|IntArr|Int|Byte|Mat3|Tensor|Vec2", "data mapped onto spheres");
    p_normalsOut = addOutputPort("normals", "Vec3", "normals for the spheres");

    p_radius = addFloatParam("radius", "Base radius of the sphere, used if input radius is not available");
    p_radius->setValue(0.1f);

    p_scale = addFloatParam("scale", "Scale factor to apply on the data");
    p_scale->setValue(1.0f);

    // keep in sync with OpenCOVER/kernel/coSphere.cpp, ...
    const char *def_modes[] = { "Polygons", "CPU Billboards", "Cg Shader", "ARB Point Sprites", "Particle Cloud", "Disc", "Textured square", "Cg Shader Inverted" };
    m_pRenderMethod = addChoiceParam("render_method", "Render method for sphere rendering");
    m_pRenderMethod->setValue(sizeof(def_modes) / sizeof(def_modes[0]), def_modes, 0);
}

int Sphere::compute(const char *)
{
    //input data structures
    const coDoPoints *points = NULL;
    const coDoFloat *sdata_in = NULL;
    const coDoVec3 *vdata_in = NULL;

    //output data structures
    coDoTriangleStrips *spheres = NULL;
    coDoVec3 *normals = NULL;
    coDoSpheres *spheresBillboards = NULL;

    // the data
    int numPoints;
    float *xPoints, *yPoints, *zPoints;
    float *dataIn[3];

    float *xSpheres, *ySpheres, *zSpheres;
    float *normalsOut[3];
    float *dataOut = NULL;
    int *vl, *tsl;

    int hasData;
    float radius, scale;

    int i = 0;

    float *fScoordx = NULL, *fScoordy = NULL, *fScoordz = NULL, *fSradii = NULL;

    // assemble output
    const coDistributedObject *obj1 = p_pointsIn->getCurrentObject();
    const coDistributedObject *obj2 = p_dataIn->getCurrentObject();
    const coDistributedObject *obj3 = p_colorsIn->getCurrentObject();

    // get the points first
    if (!obj1)
    {
        sendError("Did not receive object at port '%s'", p_pointsIn->getName());
        return FAIL;
    }
    else if ((points = dynamic_cast<const coDoPoints *>(obj1)))
    {
        numPoints = points->getNumPoints();
        points->getAddresses(&xPoints, &yPoints, &zPoints);
    }
    else if (const coDoStructuredGrid *sgrid_in = dynamic_cast<const coDoStructuredGrid *>(obj1))
    {
        int x_size, y_size, z_size;
        sgrid_in->getAddresses(&xPoints, &yPoints, &zPoints);
        sgrid_in->getGridSize(&x_size, &y_size, &z_size);
        numPoints = x_size * y_size * z_size;
    }
    else if (const coDoUnstructuredGrid *grid_in = dynamic_cast<const coDoUnstructuredGrid *>(obj1))
    {
        int *el, *cl, numelem, numconn;
        grid_in->getAddresses(&el, &cl, &xPoints, &yPoints, &zPoints);
        grid_in->getGridSize(&numelem, &numconn, &numPoints);
    }
    else
    {
        sendError("Received illegal type at port '%s'", p_pointsIn->getName());
        return FAIL;
    }

    // and see if we have any data or colors
    hasData = 0;
    dataIn[0] = dataIn[1] = dataIn[2] = NULL;

    if (obj2)
    {
        // we have data
        if ((sdata_in = dynamic_cast<const coDoFloat *>(obj2)))
        {
            sdata_in->getAddress(&dataIn[0]);
            int no_points = sdata_in->getNumPoints();
            if (no_points == numPoints)
            {
                hasData = 1;
            }
            else if (no_points != 0)
            {
                sendError("Input scaling data (%d) does not match input grid (%d)", no_points, numPoints);
                return FAIL;
            }
        }
        else if ((vdata_in = dynamic_cast<const coDoVec3 *>(obj2)))
        {
            vdata_in->getAddresses(&dataIn[0], &dataIn[1], &dataIn[2]);
            int no_points = vdata_in->getNumPoints();
            if (no_points == numPoints)
            {
                hasData = 2;
            }
            else if (no_points != 0)
            {
                sendError("Input scaling data (%d) does not match input grid (%d)", no_points, numPoints);
                return FAIL;
            }
        }
        else
        {
            sendError("Received illegal type at port '%s'", p_dataIn->getName());
            return FAIL;
        }
    }

    // and get the parameters
    scale = p_scale->getValue();
    radius = p_radius->getValue();

    // build output objects
    if (m_pRenderMethod->getValue() == 0)
    {
        coObjInfo inf = p_spheresOut->getNewObjectInfo();
        points->copyObjInfo(&inf);
        spheres = new coDoTriangleStrips(inf, coSphere::TESSELATION * numPoints, 42 * numPoints, 6 * numPoints);
        spheres->getAddresses(&xSpheres, &ySpheres, &zSpheres, &vl, &tsl);

        normals = new coDoVec3(p_normalsOut->getObjName(), coSphere::TESSELATION * numPoints);
        normals->getAddresses(&normalsOut[0], &normalsOut[1], &normalsOut[2]);

        coSphere *mySphere = new coSphere();
        mySphere->computeSpheres(scale, radius, dataIn,
                                 xSpheres, ySpheres, zSpheres,
                                 xPoints, yPoints, zPoints,
                                 normalsOut, tsl, vl, numPoints,
                                 dataOut, NULL, 0, hasData);
        p_spheresOut->setCurrentObject(spheres);
        p_normalsOut->setCurrentObject(normals);

        if (obj3)
        {
            coDistributedObject *mapOutput = mySphere->amplifyData(obj3, p_dataOut->getObjName(), numPoints);
            if (!mapOutput)
                sendWarning("Data at input port '%s' is not supported", p_colorsIn->getName());
            p_dataOut->setCurrentObject(mapOutput);
        }
    }
    else if (m_pRenderMethod->getValue() > 0)
    {
        coObjInfo inf = p_spheresOut->getNewObjectInfo();
        points->copyObjInfo(&inf);
        spheresBillboards = new coDoSpheres(inf, numPoints, xPoints, yPoints, zPoints);
        spheresBillboards->getAddresses(&fScoordx, &fScoordy, &fScoordz, &fSradii);
        const char *rm = "CPU_BILLBOARDS";
        switch (m_pRenderMethod->getValue())
        {
        case 2:
            rm = "CG_SHADER";
            break;
        case 3:
            rm = "POINT_SPRITES";
            break;
        case 4:
            rm = "PARTICLE_CLOUD";
            break;
        case 5:
            rm = "DISC";
            break;
        case 6:
            rm = "TEXTURE";
            break;
        case 7:
            rm = "CG_SHADER_INVERTED";
            break;
        }
        spheresBillboards->addAttribute("RENDER_METHOD", rm);
        if (obj3)
        {
#ifndef YAC
            /* zu mappende Daten werden durchgeschleust für jeden der möglichen Datentypen*/
            coDistributedObject *mapOutput = obj3->clone(p_dataOut->getObjName());
            if (!mapOutput)
                sendWarning("Data at input port '%s' is not supported", p_colorsIn->getName());
            p_dataOut->setCurrentObject(mapOutput);
#else
            p_dataOut->setCurrentObject(obj3);
#endif
        }

        if (hasData)
        {
            for (i = 0; i < numPoints; i++)
            {
                fSradii[i] = (dataIn[0])[i] * scale;
            }
        }
        else
        {
            for (i = 0; i < numPoints; i++)
            {
                fSradii[i] = radius;
            }
        }
        p_spheresOut->setCurrentObject(spheresBillboards);
    }

    // done
    return SUCCESS;
}

MODULE_MAIN(Tools, Sphere)
