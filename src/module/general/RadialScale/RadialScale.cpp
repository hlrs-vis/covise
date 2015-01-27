/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\
**                                                                        **
** Description: Rescale coordinates (and data) according to radial        **
**              distance to a center                                      **
**                                                                        **
**      Author: Martin Aumueller (aumueller@uni-koeln.de)                 **
**                                                                        **
\**************************************************************************/

#include <do/coDoData.h>
#include <do/coDoUniformGrid.h>
#include <do/coDoRectilinearGrid.h>
#include <do/coDoStructuredGrid.h>
#include <do/coDoUnstructuredGrid.h>
#include <do/coDoSpheres.h>
#include <api/coSimpleModule.h>

using namespace covise;

class RadialScale : public coSimpleModule
{
private:
    // Ports:
    coInputPort *piGrid;
    coInputPort *piData;

    coOutputPort *poGrid;
    coOutputPort *poData;

    coChoiceParam *pRescaleFunction;
    coFloatVectorParam *pCenter;
    coFloatParam *pFixedRadius;

    // Methods:
    virtual int compute(const char *port);

public:
    RadialScale(int argc, char *argv[]);
};

RadialScale::RadialScale(int argc, char *argv[])
    : coSimpleModule(argc, argv, "Rescale radial component of polar coordinates")
{

    // Create ports:
    piGrid = addInputPort("GridIn0", "Points|Spheres", "input points or spheres");
    piData = addInputPort("DataIn0", "Float", "input data to be rescaled accordingly");
    poGrid = addOutputPort("GridOut0", "Points|Spheres", "output points or spheres");
    poData = addOutputPort("DataOut0", "Float", "rescaled data");

    const char *mapping[] = { "Identity", "Invert", "Square root", "Log(1+...)" };
    pRescaleFunction = addChoiceParam("Mapping", "mapping function");
    pRescaleFunction->setValue(4, mapping, 1);
    pCenter = addFloatVectorParam("Center", "center");
    pCenter->setValue(0., 0., 0.);
    pFixedRadius = addFloatParam("FixedRadius", "radius that will be mapped onto itself");
    pFixedRadius->setValue(1.0);
}

/// Compute routine: load checkpoint file
int RadialScale::compute(const char *)
{
    float c[3];
    pCenter->getValue(c[0], c[1], c[2]);
    float rfix = pFixedRadius->getValue();
    float rfix2 = rfix * rfix;
    float rscale = (powf(2.f, rfix) - 1.f) / rfix;
    int func = pRescaleFunction->getValue();

    const coDistributedObject *inGrid = piGrid->getCurrentObject();
    if (!inGrid)
    {
        Covise::sendError("did not receive input data");
        return STOP_PIPELINE;
    }

    coDoFloat *outData = NULL;
    float *d = NULL;
    if (const coDoFloat *inData = dynamic_cast<const coDoFloat *>(piData->getCurrentObject()))
    {
        outData = static_cast<coDoFloat *>(inData->clone(poData->getObjName()));
        d = outData->getAddress();
    }

    coDistributedObject *outGrid = NULL;
    float *x = NULL, *y = NULL, *z = NULL;
    int numPoints = 0;
    if (const coDoPoints *points = dynamic_cast<const coDoPoints *>(inGrid))
    {
        coDoPoints *p = static_cast<coDoPoints *>(points->clone(poGrid->getObjName()));
        outGrid = p;
        p->getAddresses(&x, &y, &z);
        numPoints = p->getNumPoints();
    }
    else if (const coDoSpheres *spheres = dynamic_cast<const coDoSpheres *>(inGrid))
    {
        coDoSpheres *s = static_cast<coDoSpheres *>(spheres->clone(poGrid->getObjName()));
        outGrid = s;
        float *r;
        s->getAddresses(&x, &y, &z, &r);
        numPoints = s->getNumSpheres();
    }

    const float invln2 = 1.f / logf(2.f);
    if (func)
    {
        for (int i = 0; i < numPoints; ++i)
        {
            float dx = x[i] - c[0];
            float dy = y[i] - c[1];
            float dz = z[i] - c[2];
            float r = sqrt(dx * dx + dy * dy + dz * dz);
            float s = 1.;
            if (r > 0.)
                s = func == 1 ? rfix2 / r : func == 2 ? sqrt(r) : logf(1.f + r * rscale) * invln2;
            x[i] = s * dx + c[0];
            y[i] = s * dy + c[1];
            z[i] = s * dz + c[2];
            if (d)
                d[i] *= s;
        }
    }

    poGrid->setCurrentObject(outGrid);
    poData->setCurrentObject(outData);

    return CONTINUE_PIPELINE;
}

MODULE_MAIN(Converter, RadialScale)
