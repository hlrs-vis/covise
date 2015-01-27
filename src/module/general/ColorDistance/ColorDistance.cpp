/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\ 
 **                                                        (C)2005 RRZK  **
 ** Description:   Simple calculations on volume data.                   **
 **                                                                      **
 ** Author:        Martin Aumueller <aumueller@uni-koeln.de>             **
 **                Sandra Meid <sandra.meid@uni-koeln.de>                **
 **                                                                      **
 ** Creation Date: 05.01.2005                                            **
\**************************************************************************/

#include <api/coModule.h>
#include "ColorDistance.h"
#include <do/coDoData.h>

#ifdef BYTESWAP
#define byteSwap(x) (void)(x)
#endif

/// constructor
coColorDistance::coColorDistance(int argc, char *argv[])
    : coSimpleModule(argc, argv, "Compute distance to reference color")
{
    // Create ports:
    piR = addInputPort("Red", "Float", "Scalar volume data (red channel)");
    piR->setInfo("Scalar volume data (red channel)");

    piG = addInputPort("Green", "Float", "Scalar volume data (green channel)");
    piG->setInfo("Scalar volume data (green channel)");

    piB = addInputPort("Blue", "Float", "Scalar volume data (blue channel)");
    piB->setInfo("Scalar volume data (blue channel)");

    poVolume = addOutputPort("Data", "Float", "Scalar volume data");
    poVolume->setInfo("Scalar volume data (range 0-1)");

    // Create parameters:
    paReferenceColor = addColorParam("ReferenceColor", "Color to which the distance is calculated");
    paReferenceColor->setValue(0.0, 0.0, 0.0, 1);

    const char *metric[] = { "euclidian distance", "manhattan distance" };
    paMetric = addChoiceParam("Metric", "Metric for calculation of the distance for transparent values.");
    paMetric->setValue(2, metric, 0);

    paSlider = addFloatSliderParam("DistanceBase", "This value is added to the calculated distance.");
    paSlider->setValue(-10, 10, 1);

    paSlider2 = addFloatSliderParam("DistanceMultiplier", "This value multiplies the calculated distance.");
    paSlider2->setValue(-10, 10, 1);
}

/// compute-routine
int coColorDistance::compute(const char *)
{
    float *r, *g, *b;

    const coDoFloat *red = dynamic_cast<const coDoFloat *>(piR->getCurrentObject());
    const coDoFloat *green = dynamic_cast<const coDoFloat *>(piG->getCurrentObject());
    const coDoFloat *blue = dynamic_cast<const coDoFloat *>(piB->getCurrentObject());
    if (!red || !green || !blue)
    {
        sendError("Data type error: must be a structured scalar data");
        return STOP_PIPELINE;
    }

    long gridSize = red->getNumPoints();
    red->getAddress(&r);
    green->getAddress(&g);
    blue->getAddress(&b);

    float rr = paReferenceColor->getValue(0);
    float rg = paReferenceColor->getValue(1);
    float rb = paReferenceColor->getValue(2);

    coDoFloat *volumeOut = new coDoFloat(poVolume->getObjName(), gridSize);
    float *data = NULL;
    volumeOut->getAddress(&data);

    for (long i = 0; i < gridSize; ++i)
    {
        if (paMetric->getValue() == 0)
        {
            data[i] = paSlider2->getValue() + paSlider->getValue() * (sqrtf((r[i] - rr) * (r[i] - rr) + (g[i] - rg) * (g[i] - rg) + (b[i] - rb) * (b[i] - rb)));
            if (data[i] < 0.f)
                data[i] = 0.f;
        }

        if (paMetric->getValue() == 1)
        {
            data[i] = paSlider2->getValue() + paSlider->getValue() * (fabs(r[i] - rr) + fabs(g[i] - rg) + fabs(b[i] - rb));
            if (data[i] < 0.f)
                data[i] = 0.f;
            if (r[i] > .4 && g[i] > .3 && b[i] > .4)
                data[i] = 0.f;
            if (r[i] < 0.01 && g[i] < 0.01 && b[i] < 0.01)
                data[i] = 0.f;
        }
    }

    poVolume->setCurrentObject(volumeOut);

    return CONTINUE_PIPELINE;
}

MODULE_MAIN(Filter, coColorDistance)
