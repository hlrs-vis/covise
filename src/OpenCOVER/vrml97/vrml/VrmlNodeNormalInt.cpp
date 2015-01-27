/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeNormalInt.cpp

#include "VrmlNodeNormalInt.h"
#include "VrmlNodeType.h"
#include "MathUtils.h"

#include "VrmlScene.h"

using namespace vrml;

// NormalInt factory.

static VrmlNode *creator(VrmlScene *scene)
{
    return new VrmlNodeNormalInt(scene);
}

// Define the built in VrmlNodeType:: "NormalInterpolator" fields

VrmlNodeType *VrmlNodeNormalInt::defineType(VrmlNodeType *t)
{
    static VrmlNodeType *st = 0;

    if (!t)
    {
        if (st)
            return st; // Only define the type once.
        t = st = new VrmlNodeType("NormalInterpolator", creator);
    }

    VrmlNodeChild::defineType(t); // Parent class
    t->addEventIn("set_fraction", VrmlField::SFFLOAT);
    t->addExposedField("key", VrmlField::MFFLOAT);
    t->addExposedField("keyValue", VrmlField::MFVEC3F);
    t->addEventOut("value_changed", VrmlField::MFVEC3F);

    return t;
}

VrmlNodeType *VrmlNodeNormalInt::nodeType() const { return defineType(0); }

VrmlNodeNormalInt::VrmlNodeNormalInt(VrmlScene *scene)
    : VrmlNodeChild(scene)
{
}

VrmlNodeNormalInt::~VrmlNodeNormalInt()
{
}

VrmlNode *VrmlNodeNormalInt::cloneMe() const
{
    return new VrmlNodeNormalInt(*this);
}

std::ostream &VrmlNodeNormalInt::printFields(std::ostream &os, int indent)
{
    if (d_key.size() > 0)
        PRINT_FIELD(key);
    if (d_keyValue.size() > 0)
        PRINT_FIELD(keyValue);

    return os;
}

void VrmlNodeNormalInt::eventIn(double timeStamp,
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

        int nNormals = d_keyValue.size() / d_key.size();
        int n = d_key.size() - 1;

        if (f < d_key[0])
        {
            d_value.set(nNormals, d_keyValue[0]);
        }
        else if (f > d_key[n])
        {
            d_value.set(nNormals, d_keyValue[n * nNormals]);
        }
        else
        {
            // Reserve enough space for the new value
            d_value.set(nNormals, 0);

            for (int i = 0; i < n; ++i)
                if (d_key[i] <= f && f <= d_key[i + 1])
                {
                    float *v1 = d_keyValue[i * nNormals];
                    float *v2 = d_keyValue[(i + 1) * nNormals];
                    float *x = d_value.get();

                    f = (f - d_key[i]) / (d_key[i + 1] - d_key[i]);

                    // Interpolate on the surface of unit sphere.
                    // Contributed by S. K. Bose. (bose@garuda.barc.ernet.in)
                    for (int j = 0; j < nNormals; ++j)
                    {
                        //*x++ = v1[0] + f * (v2[0] - v1[0]);
                        //*x++ = v1[1] + f * (v2[1] - v1[1]);
                        //*x++ = v1[2] + f * (v2[2] - v1[2]);
                        float alpha, beta;
                        float dotval = v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
                        if ((dotval + 1.0) > FPTOLERANCE) // Vectors are opposite
                        {
                            if ((1.0 - dotval) > FPTOLERANCE) // Vectors coincide
                            {
                                float omega = acos(dotval);
                                float sinomega = sin(omega);
                                alpha = sin((1.0f - f) * omega) / sinomega;
                                beta = sin(f * omega) / sinomega;
                            }
                            else
                            {
                                // Do linear interpolation...
                                alpha = 1.0f - f;
                                beta = f;
                            }
                        }
                        else
                        { // Do linear interpolation...
                            alpha = 1.0f - f;
                            beta = f;
                        }
                        *x++ = alpha * v1[0] + beta * v2[0];
                        *x++ = alpha * v1[1] + beta * v2[1];
                        *x++ = alpha * v1[2] + beta * v2[2];

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

// Set the value of one of the node fields.

void VrmlNodeNormalInt::setField(const char *fieldName,
                                 const VrmlField &fieldValue)
{
    if
        TRY_FIELD(key, MFFloat)
    else if
        TRY_FIELD(keyValue, MFVec3f)
    else
        VrmlNodeChild::setField(fieldName, fieldValue);
}

const VrmlField *VrmlNodeNormalInt::getField(const char *fieldName) const
{
    if (strcmp(fieldName, "key") == 0)
        return &d_key;
    else if (strcmp(fieldName, "keyValue") == 0)
        return &d_keyValue;

    return VrmlNodeChild::getField(fieldName);
}
