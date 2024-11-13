/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeCoordinateInt.cpp

#include "VrmlNodeCoordinateInt.h"
#include "VrmlNodeType.h"

#include "VrmlScene.h"
#include "System.h"

using namespace vrml;

void VrmlNodeCoordinateInt::initFields(VrmlNodeCoordinateInt *node, VrmlNodeType *t)
{
    VrmlNodeChild::initFields(node, t); // Parent class
    initFieldsHelper(node, t,
                     exposedField("key", node->d_key),
                     exposedField("keyValue", node->d_keyValue));
    if (t)
    {
        t->addEventIn("set_fraction", VrmlField::SFFLOAT);
        t->addEventOut("value_changed", VrmlField::MFVEC3F);
    }
}

const char *VrmlNodeCoordinateInt::name() { return "CoordinateInterpolator"; }


VrmlNodeCoordinateInt::VrmlNodeCoordinateInt(VrmlScene *scene)
    : VrmlNodeChild(scene, name())
{
}

void VrmlNodeCoordinateInt::eventIn(double timeStamp,
                                    const char *eventName,
                                    const VrmlField *fieldValue)
{
    if (strcmp(eventName, "set_fraction") == 0)
    {
        if (!fieldValue->toSFFloat())
        {
            System::the->error("Invalid type for %s eventIn %s (expected SFFloat).\n",
                               nodeType()->getName(), eventName);
            return;
        }
        float f = fieldValue->toSFFloat()->get();

        int nCoords = d_keyValue.size() / d_key.size();
        int n = d_key.size() - 1;

        if (f < d_key[0])
        {
            d_value.set(nCoords, d_keyValue[0]);
        }
        else if (f > d_key[n])
        {
            d_value.set(nCoords, d_keyValue[n * nCoords]);
        }
        else
        {
            // Reserve enough space for the new value
            d_value.set(nCoords, 0);

            for (int i = 0; i < n; ++i)
                if (d_key[i] <= f && f <= d_key[i + 1])
                {
                    float *v1 = d_keyValue[i * nCoords];
                    float *v2 = d_keyValue[(i + 1) * nCoords];
                    float *x = d_value.get();

                    f = (f - d_key[i]) / (d_key[i + 1] - d_key[i]);

                    for (int j = 0; j < nCoords; ++j)
                    {
                        *x++ = v1[0] + f * (v2[0] - v1[0]);
                        *x++ = v1[1] + f * (v2[1] - v1[1]);
                        *x++ = v1[2] + f * (v2[2] - v1[2]);
                        v1 += 3;
                        v2 += 3;
                    }

                    break;
                }
        }

        // Send the new value
        eventOut(timeStamp, "value_changed", d_value);
    }

    // Check exposedFields
    else
    {
        VrmlNode::eventIn(timeStamp, eventName, fieldValue);

        // This node is not renderable, so don't re-render on changes to it.
        clearModified();
    }
}
