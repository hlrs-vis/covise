/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeBooleanSequencer.cpp

#include "VrmlNodeBooleanSequencer.h"
#include "VrmlNodeType.h"

#include "VrmlScene.h"
#include "VrmlSFBool.h"

using namespace vrml;

// BooleanSequencer factory.

static VrmlNode *creator(VrmlScene *scene)
{
    return new VrmlNodeBooleanSequencer(scene);
}

// Define the built in VrmlNodeType:: "BooleanSequencer" fields

VrmlNodeType *VrmlNodeBooleanSequencer::defineType(VrmlNodeType *t)
{
    static VrmlNodeType *st = 0;

    if (!t)
    {
        if (st)
            return st; // Only define the type once.
        t = st = new VrmlNodeType("BooleanSequencer", creator);
    }

    VrmlNodeChild::defineType(t); // Parent class
    t->addEventIn("next", VrmlField::SFBOOL);
    t->addEventIn("previous", VrmlField::SFBOOL);
    t->addEventIn("set_fraction", VrmlField::SFFLOAT);
    t->addEventIn("set_fraction", VrmlField::SFFLOAT);
    t->addExposedField("key", VrmlField::MFFLOAT);
    t->addExposedField("keyValue", VrmlField::MFBOOL);
    t->addEventOut("value_changed", VrmlField::SFBOOL);

    return t;
}

VrmlNodeType *VrmlNodeBooleanSequencer::nodeType() const { return defineType(0); }

VrmlNodeBooleanSequencer::VrmlNodeBooleanSequencer(VrmlScene *scene)
    : VrmlNodeChild(scene)
{
}

VrmlNodeBooleanSequencer::~VrmlNodeBooleanSequencer()
{
}

VrmlNode *VrmlNodeBooleanSequencer::cloneMe() const
{
    return new VrmlNodeBooleanSequencer(*this);
}

std::ostream &VrmlNodeBooleanSequencer::printFields(std::ostream &os, int indent)
{
    if (d_key.size() > 0)
        PRINT_FIELD(key);
    if (d_keyValue.size() > 0)
        PRINT_FIELD(keyValue);

    return os;
}

void VrmlNodeBooleanSequencer::eventIn(double timeStamp, const char *eventName,
                                       const VrmlField *fieldValue)
{
    bool send = true;
    int offset = 0;
    if (strcmp(eventName, "set_fraction") == 0)
    {
        if (!fieldValue->toSFFloat())
        {
            System::the->error("Invalid type for %s eventIn %s (expected SFFloat).\n",
                               nodeType()->getName(), eventName);
            return;
        }
    }
    else if (strcmp(eventName, "next") == 0)
    {
        offset = 1;
    }
    else if (strcmp(eventName, "previous") == 0)
    {
        offset = -1;
    }
    else
        send = false;

    float f = fieldValue->toSFFloat()->get();

    int n = d_key.size() - 1;
    if (n < 0 || ((n == 0) && (offset == -1)))
    {
        bool initialValue = false;
        d_value.set(initialValue);
    }
    else if (n >= 0 && f < d_key[0])
        d_value.set(d_keyValue[0]);
    else if ((f > d_key[n]) || ((f > d_key[n - 1]) && (offset == 1)))
        d_value.set(d_keyValue[n]);
    else
    {
        // should cache the last index used...
        for (int i = 0; i < n; ++i)
            if (d_key[i] <= f && f <= d_key[i + 1])
            {
                bool b = d_keyValue[i + offset];
                d_value.set(b);
                break;
            }
    }

    if (send)
    {
        // Send the new value
        eventOut(timeStamp, "value_changed", d_value);
    }
    else
    {
        // Check exposedFields
        VrmlNode::eventIn(timeStamp, eventName, fieldValue);

        // This node is not renderable, so don't re-render on changes to it.
        clearModified();
    }
}

// Set the value of one of the node fields.

void VrmlNodeBooleanSequencer::setField(const char *fieldName,
                                        const VrmlField &fieldValue)
{
    if
        TRY_FIELD(key, MFFloat)
    else if
        TRY_FIELD(keyValue, MFBool)
    else
        VrmlNodeChild::setField(fieldName, fieldValue);
}

const VrmlField *VrmlNodeBooleanSequencer::getField(const char *fieldName) const
{
    if (strcmp(fieldName, "key") == 0)
        return &d_key;
    else if (strcmp(fieldName, "keyValue") == 0)
        return &d_keyValue;

    return VrmlNodeChild::getField(fieldName);
}

VrmlNodeBooleanSequencer *VrmlNodeBooleanSequencer::toBooleanSequencer() const
{
    return (VrmlNodeBooleanSequencer *)this;
}

const VrmlMFFloat &VrmlNodeBooleanSequencer::getKey() const
{
    return d_key;
}

const VrmlMFBool &VrmlNodeBooleanSequencer::getKeyValue() const
{
    return d_keyValue;
}
