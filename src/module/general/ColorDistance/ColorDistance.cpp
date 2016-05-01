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

    const char *cs[] = { "RGB", "HSV", "Hue-Saturation", "Hue" };
    paColorSpace = addChoiceParam("ColorSpace", "Color space used for distance calculation");
    paColorSpace->setValue(3, cs, 0);

    const char *metric[] = { "euclidian distance", "manhattan distance", "maximum" };
    paMetric = addChoiceParam("Metric", "Metric for calculation of the distance for transparent values.");
    paMetric->setValue(3, metric, 0);

    paMinMax = addFloatVectorParam("MinMax", "Allowed range of distance.", 2);
    paMinMax->setValue(0, 0.);
    paMinMax->setValue(1, 1.);

    paSlider = addFloatSliderParam("DistanceBase", "This value is added to the calculated distance.");
    paSlider->setValue(-10, 10, 1);

    paSlider2 = addFloatSliderParam("DistanceMultiplier", "This value multiplies the calculated distance.");
    paSlider2->setValue(-10, 10, -1);
}

inline void rgb2hsv(float r, float g, float b, float &h, float &s, float &v)
{
    float M = std::max(r, std::max(g, b));
    float m = std::min(r, std::min(g, b));
    float c = M - m;

    v = M;
    if (c <= 0.01)
    {
        s = 0.f;
        h = 0.f;
    }
    else 
    {
        s = v/c;
        if (r == M)
        {
            h = trunc((g-b)/c*0.5f)/3.f;
        }
        else if (g == M)
        {
            h = (1.f+trunc((b-r)/c*0.5f))/3.f;
        }
        else
        {
            h = (2.f+trunc((r-g)/c*0.5f))/3.f;
        }
    }
}

/// compute-routine
int coColorDistance::compute(const char *)
{
    const coDoFloat *red = dynamic_cast<const coDoFloat *>(piR->getCurrentObject());
    const coDoFloat *green = dynamic_cast<const coDoFloat *>(piG->getCurrentObject());
    const coDoFloat *blue = dynamic_cast<const coDoFloat *>(piB->getCurrentObject());
    if (!red || !green || !blue)
    {
        sendError("Data type error: must be a structured scalar data");
        return STOP_PIPELINE;
    }

    long gridSize = red->getNumPoints();
    const float *r = red->getAddress();
    const float *g = green->getAddress();
    const float *b = blue->getAddress();

    float rr = paReferenceColor->getValue(0);
    float rg = paReferenceColor->getValue(1);
    float rb = paReferenceColor->getValue(2);

    float ref[3];
    const bool hsv = paColorSpace->getValue()!=0;
    int dim = paColorSpace->getValue()==2 ? 2 : 3;
    if (hsv)
    {
        rgb2hsv(rr, rg, rb, ref[0], ref[1], ref[2]);
    }
    else
    {
        ref[0] = rr;
        ref[1] = rg;
        ref[2] = rb;
    }

    const float base = paSlider->getValue();
    const float scale = paSlider2->getValue();
    coDoFloat *volumeOut = new coDoFloat(poVolume->getObjName(), gridSize);
    float *data = volumeOut->getAddress();
    float min = paMinMax->getValue(0);
    float max = paMinMax->getValue(1);

    for (long i = 0; i < gridSize; ++i)
    {
        float d[3];
        if (hsv)
        {
            rgb2hsv(r[i], g[i], b[i], d[0], d[1], d[2]);
        }
        else
        {
            d[0] = r[i];
            d[1] = g[i];
            d[2] = b[i];
        }
        float m = 0.f;
        for (int c=0; c<3; ++c)
        {
            d[c] = fabs(ref[c]-d[c]);
        }
        switch(paMetric->getValue())
        {
            case 0: // euclid
                m = dim==2 ? sqrtf(d[0]*d[0]+d[1]*d[1]) : sqrtf(d[0]*d[0]+d[1]*d[1]+d[2]*d[2]);
                break;
            case 1: // manhattan
                m = dim==2 ? d[0]+d[1] : d[0]+d[1]+d[2];
                break;
            case 2:
                m = dim==2 ? std::max(d[0], d[1]) : std::max(d[0], std::max(d[1], d[2]));
                break;
        }
        m *= scale;
        m += base;
        m = std::min(m, max);
        m = std::max(m, min);
        data[i] = m;
    }

    poVolume->setCurrentObject(volumeOut);

    return CONTINUE_PIPELINE;
}

MODULE_MAIN(Filter, coColorDistance)
