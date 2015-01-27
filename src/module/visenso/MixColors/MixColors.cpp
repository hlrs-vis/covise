/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include "MixColors.h"

#include <do/coDoData.h>

#define MAX(a, b) (a > b ? a : b)

MixColors::MixColors(int argc, char *argv[])
    : coSimpleModule(argc, argv, "Mix colors according to their tranparency")
{
    p_colors1 = addInputPort("Colors0", "RGBA", "colors 1");
    p_colors1->setRequired(1);

    p_colors2 = addInputPort("Colors1", "RGBA", "colors 2");
    p_colors2->setRequired(1);

    p_colorsOut = addOutputPort("Colors", "RGBA", "colors");
}

int MixColors::compute(const char *)
{

    const coDistributedObject *do1 = p_colors1->getCurrentObject();
    if (!do1)
        return FAIL;
    const coDoRGBA *colors1 = dynamic_cast<const coDoRGBA *>(do1);
    if (!colors1)
        return FAIL;

    const coDistributedObject *do2 = p_colors2->getCurrentObject();
    if (!do2)
        return FAIL;
    const coDoRGBA *colors2 = dynamic_cast<const coDoRGBA *>(do2);
    if (!colors2)
        return FAIL;

    if (colors1->getNumPoints() != colors2->getNumPoints())
        return FAIL;

    // CREATE

    int numPoints = colors1->getNumPoints();
    const char *outName = p_colorsOut->getObjName();
    coDoRGBA *outObj = new coDoRGBA(outName, numPoints);
    outObj->copyAllAttributes(colors1);

    // CALCULATE

    int *outPtr;
    outObj->getAddress(&outPtr);
    unsigned int *outPacked = (unsigned int *)outPtr;
    int *colors1Ptr;
    colors1->getAddress(&colors1Ptr);
    unsigned int *colors1Packed = (unsigned int *)colors1Ptr;
    int *colors2Ptr;
    colors2->getAddress(&colors2Ptr);
    unsigned int *colors2Packed = (unsigned int *)colors2Ptr;

    unsigned char r1, g1, b1, a1, r2, g2, b2, a2, r, g, b, a;
    for (int i = 0; i < numPoints; ++i)
    {
        r1 = (colors1Packed[i] & (255 << 24)) >> 24;
        g1 = (colors1Packed[i] & (255 << 16)) >> 16;
        b1 = (colors1Packed[i] & (255 << 8)) >> 8;
        a1 = (colors1Packed[i] & 255);
        r2 = (colors2Packed[i] & (255 << 24)) >> 24;
        g2 = (colors2Packed[i] & (255 << 16)) >> 16;
        b2 = (colors2Packed[i] & (255 << 8)) >> 8;
        a2 = (colors2Packed[i] & 255);
        if (a1 == 0)
        {
            r = r2;
            g = g2;
            b = b2;
            a = a2;
        }
        else
        {
            r = char((float(r1) * float(a1) + float(r2) * float(a2)) / (float(a1) * float(a2)));
            g = char((float(g1) * float(a1) + float(g2) * float(a2)) / (float(a1) * float(a2)));
            b = char((float(b1) * float(a1) + float(b2) * float(a2)) / (float(a1) * float(a2)));
            a = MAX(a1, a2);
        }
        outPacked[i] = (r << 24) | (g << 16) | (b << 8) | a;
    }

    p_colorsOut->setCurrentObject(outObj);

    return SUCCESS;
}

MODULE_MAIN(Tools, MixColors)
