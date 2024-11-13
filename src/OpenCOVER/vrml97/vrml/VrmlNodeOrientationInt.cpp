/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeOrientationInt.cpp

#include "VrmlNodeOrientationInt.h"
#include "VrmlNodeType.h"

#include "VrmlScene.h"
#include "System.h"
#include <math.h>

using namespace vrml;

// OrientationInt factory.

static VrmlNode *creator(VrmlScene *scene)
{
    return new VrmlNodeOrientationInt(scene);
}

// Define the built in VrmlNodeType:: "OrientationInt" fields
void VrmlNodeOrientationInt::initFields(VrmlNodeOrientationInt *node, VrmlNodeType *t)
{
    VrmlNodeChild::initFields(node, t); // Parent class
    initFieldsHelper(node, t,
                     exposedField("key", node->d_key),
                     exposedField("keyValue", node->d_keyValue));
    if (t)
    {
        t->addEventIn("set_fraction", VrmlField::SFFLOAT);
        t->addEventOut("value_changed", VrmlField::SFROTATION);
    }
}

const char *VrmlNodeOrientationInt::name() { return "OrientationInterpolator"; }

VrmlNodeOrientationInt::VrmlNodeOrientationInt(VrmlScene *scene)
    : VrmlNodeChild(scene, name())
{
}

void VrmlNodeOrientationInt::eventIn(double timeStamp,
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

        //printf("OI.set_fraction %g ", f);

        int n = d_key.size() - 1;
        if (n < 0)
        {
            // http://web3d.org/x3d/specifications/vrml/ISO-IEC-14772-VRML97/part1/concepts.html#4.6.8
            // ... Results are undefined if the number of values in the key
            // field of an interpolator is not the same as the number of values
            // in the keyValue field.
            // ... If keyValue is empty (i.e., [ ]), the initial value for the
            // eventOut type is returned (e.g., (0, 0, 0) for SFVec3f);
            VrmlSFRotation initialValue;
            d_value.set(initialValue.x(), initialValue.y(), initialValue.z(),
                        initialValue.r());
        }
        else if (f < d_key[0])
        {
            float *v0 = d_keyValue[0];
            //printf(" 0 [%g %g %g %g]\n", v0[0], v0[1], v0[2], v0[3] );
            d_value.set(v0[0], v0[1], v0[2], v0[3]);
        }
        else if (f > d_key[n])
        {
            float *vn = d_keyValue[n];
            //printf(" n [%g %g %g %g]\n", vn[0], vn[1], vn[2], vn[3] );
            d_value.set(vn[0], vn[1], vn[2], vn[3]);
        }
        else
        {
            for (int i = 0; i < n; ++i)
                if (d_key[i] <= f && f <= d_key[i + 1])
                {
                    float *v1 = d_keyValue[i];
                    float *v2 = d_keyValue[i + 1];

                    // Interpolation factor
                    f = (f - d_key[i]) / (d_key[i + 1] - d_key[i]);

                    float x, y, z, r1, r2;
                    r1 = v1[3];
                    if (v2[3] == 0.0)
                    {
                        x = v1[0];
                        y = v1[1];
                        z = v1[2];
                        r2 = v2[3];
                    }
                    else if (v1[3] == 0.0)
                    {
                        x = v2[0];
                        y = v2[1];
                        z = v2[2];
                        r2 = v2[3];
                    }
                    else
                    {
                        // Make sure the vectors are not pointing opposite ways
                        if (v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2] < 0.0)
                        {
                            x = v1[0] + f * (-v2[0] - v1[0]);
                            y = v1[1] + f * (-v2[1] - v1[1]);
                            z = v1[2] + f * (-v2[2] - v1[2]);
                            r2 = -v2[3];
                        }
                        else
                        {
                            x = v1[0] + f * (v2[0] - v1[0]);
                            y = v1[1] + f * (v2[1] - v1[1]);
                            z = v1[2] + f * (v2[2] - v1[2]);
                            r2 = v2[3];
                        }
                    }
                    // Interpolate angles via the shortest direction
                    if (fabs(r2 - r1) > M_PI)
                    {
                        if (r2 > r1)
                            r1 += (float)(2.0f * M_PI);
                        else
                            r2 += (float)(2.0f * M_PI);
                    }
                    float r = r1 + f * (r2 - r1);
                    if (r >= 2.0 * M_PI)
                        r -= (float)(2.0f * M_PI);
                    else if (r < 0.0)
                        r += (float)(2.0f * M_PI);

                    //printf(" %g between (%d,%d) [%g %g %g %g]\n", f, i, i+1,
                    //x, y, z, r);

                    d_value.set(x, y, z, r);
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

const VrmlMFFloat &VrmlNodeOrientationInt::getKey() const
{
    return d_key;
}

const VrmlMFRotation &VrmlNodeOrientationInt::getKeyValue() const
{
    return d_keyValue;
}
