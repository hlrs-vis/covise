/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeColorInt.cpp

#include "VrmlNodeColorInt.h"
#include "VrmlNodeType.h"

#include "VrmlScene.h"
#include "System.h"

#include <math.h>

using namespace vrml;

// ColorInterpolator factory.

static VrmlNode *creator(VrmlScene *scene)
{
    return new VrmlNodeColorInt(scene);
}

void VrmlNodeColorInt::initFields(VrmlNodeColorInt *node, VrmlNodeType *t)
{
    initFieldsHelper(node, t,
                     exposedField("key", node->d_key),
                     exposedField("keyValue", node->d_keyValue));
    if(t)
    {
        t->addEventIn("set_fraction", VrmlField::SFFLOAT);
        t->addEventOut("value_changed", VrmlField::SFCOLOR);
    }
    VrmlNodeChild::initFields(node, t);
}

const char *VrmlNodeColorInt::name() { return "ColorInterpolator"; }

VrmlNodeColorInt::VrmlNodeColorInt(VrmlScene *scene)
    : VrmlNodeChild(scene, name())
{
}

void VrmlNodeColorInt::eventIn(double timeStamp,
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

        int n = d_key.size() - 1;
        if (n < 0)
        {
            // http://web3d.org/x3d/specifications/vrml/ISO-IEC-14772-VRML97/part1/concepts.html#4.6.8
            // ... Results are undefined if the number of values in the key
            // field of an interpolator is not the same as the number of values
            // in the keyValue field.
            // ... If keyValue is empty (i.e., [ ]), the initial value for the
            // eventOut type is returned
            VrmlSFColor initialValue;
            d_value.set(initialValue.r(), initialValue.g(), initialValue.b());
        }
        else if (f < d_key[0])
            d_value.set(d_keyValue[0][0], d_keyValue[0][1], d_keyValue[0][2]);
        else if (f > d_key[n])
            d_value.set(d_keyValue[n][0], d_keyValue[n][1], d_keyValue[n][2]);
        else
        {
            //  convert to HSV for the interpolation...
            for (int i = 0; i < n; ++i)
                if (d_key[i] <= f && f <= d_key[i + 1])
                {
                    float *rgb1 = d_keyValue[i];
                    float *rgb2 = d_keyValue[i + 1];

                    f = (f - d_key[i]) / (d_key[i + 1] - d_key[i]);
                    float h1, s1, v1, h2, s2, v2;
                    VrmlSFColor::RGBtoHSV(rgb1[0], rgb1[1], rgb1[2],
                                          h1, s1, v1);
                    VrmlSFColor::RGBtoHSV(rgb2[0], rgb2[1], rgb2[2],
                                          h2, s2, v2);

                    // Interpolate angles via the shortest direction
                    if (fabs(h2 - h1) > 180.0)
                    {
                        if (h2 > h1)
                            h1 += 360.0;
                        else
                            h2 += 360.0;
                    }
                    float h = h1 + f * (h2 - h1);
                    if (h >= 360.0)
                        h -= 360.0;
                    else if (h < 0.0)
                        h += 360.0;

                    float s = s1 + f * (s2 - s1);
                    float v = v1 + f * (v2 - v1);
                    float r, g, b;
                    VrmlSFColor::HSVtoRGB(h, s, v, r, g, b);
                    d_value.set(r, g, b);
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
