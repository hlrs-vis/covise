/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeNavigationInfo.cpp

#include "VrmlNodeNavigationInfo.h"

#include "MathUtils.h"
#include "VrmlNodeType.h"
#include "VrmlScene.h"

using std::cerr;
using std::endl;
using namespace vrml;

//  NavigationInfo factory.
//  Since NavInfo is a bindable child node, the first one created needs
//  to notify its containing scene.

static VrmlNode *creator(VrmlScene *scene)
{
    return new VrmlNodeNavigationInfo(scene);
}

// Define the built in VrmlNodeType:: "NavigationInfo" fields

VrmlNodeType *VrmlNodeNavigationInfo::defineType(VrmlNodeType *t)
{
    static VrmlNodeType *st = 0;

    if (!t)
    {
        if (st)
            return st;
        t = st = new VrmlNodeType("NavigationInfo", creator);
    }

    VrmlNodeChild::defineType(t); // Parent class
    t->addEventIn("set_bind", VrmlField::SFBOOL);
    t->addEventIn("set_bindLast", VrmlField::SFBOOL);
    t->addExposedField("avatarSize", VrmlField::MFFLOAT);
    t->addExposedField("headlight", VrmlField::SFBOOL);
    t->addExposedField("speed", VrmlField::SFFLOAT);
    t->addExposedField("scale", VrmlField::SFFLOAT);
    t->addExposedField("near", VrmlField::SFFLOAT);
    t->addExposedField("far", VrmlField::SFFLOAT);
    t->addExposedField("type", VrmlField::MFSTRING);
    t->addExposedField("transitionType", VrmlField::MFSTRING);
    t->addExposedField("transitionTime", VrmlField::MFFLOAT);
    t->addExposedField("visibilityLimit", VrmlField::SFFLOAT);
    t->addEventOut("isBound", VrmlField::SFBOOL);

    return t;
}

VrmlNodeType *VrmlNodeNavigationInfo::nodeType() const { return defineType(0); }

VrmlNodeNavigationInfo::VrmlNodeNavigationInfo(VrmlScene *scene)
    : VrmlNodeChild(scene)
    , lastBind(true)
    , d_headlight(true)
    , d_scale(-1.0)
    , d_speed(1.0)
    , d_near(-1.0)
    , d_far(-1.0)
    , d_visibilityLimit(0.0)
{
    float avatarSize[] = { 0.25f, 1.6f, 0.75f };
    float transitionTime[] = { 1.0f };
    const char *type[] = { "WALK", "ANY" };
    const char *transitionType[] = { "ANIMATE" };

    d_transitionType.set(1, transitionType);
    d_transitionTime.set(1, transitionTime);
    d_avatarSize.set(3, avatarSize);
    d_type.set(2, type);
    if (d_scene)
        d_scene->addNavigationInfo(this);
}

VrmlNodeNavigationInfo::~VrmlNodeNavigationInfo()
{
    if (d_scene)
        d_scene->removeNavigationInfo(this);
}

VrmlNode *VrmlNodeNavigationInfo::cloneMe() const
{
    return new VrmlNodeNavigationInfo(*this);
}

VrmlNodeNavigationInfo *VrmlNodeNavigationInfo::toNavigationInfo() const
{
    return (VrmlNodeNavigationInfo *)this;
}

void VrmlNodeNavigationInfo::addToScene(VrmlScene *s, const char *)
{
    if (d_scene != s && (d_scene = s) != 0)
        d_scene->addNavigationInfo(this);
}

std::ostream &VrmlNodeNavigationInfo::printFields(std::ostream &os, int indent)
{
    if (d_avatarSize.size() != 3 || !FPEQUAL(d_avatarSize[0], 0.25) || !FPEQUAL(d_avatarSize[1], 1.6) || !FPEQUAL(d_avatarSize[2], 0.75))
        PRINT_FIELD(avatarSize);
    if (!d_headlight.get())
        PRINT_FIELD(headlight);
    if (!FPEQUAL(d_speed.get(), 1.0))
        PRINT_FIELD(speed);
    if (d_type.size() != 2 || strcmp(d_type[0], "WALK") != 0 || strcmp(d_type[1], "ANY") != 0)
        PRINT_FIELD(type);
    if (!FPZERO(d_visibilityLimit.get()))
        PRINT_FIELD(visibilityLimit);

    return os;
}

void VrmlNodeNavigationInfo::eventIn(double timeStamp,
                                     const char *eventName,
                                     const VrmlField *fieldValue)
{
    if ((strcmp(eventName, "set_bind") == 0) || (strcmp(eventName, "set_bindLast") == 0))
    {
        if (strcmp(eventName, "set_bindLast") == 0)
            lastBind = true;
        else
            lastBind = false;
        VrmlNodeNavigationInfo *current = d_scene->bindableNavigationInfoTop();
        const VrmlSFBool *b = fieldValue->toSFBool();

        if (!b)
        {
            cerr << "Error: invalid value for NavigationInfo::set_bind eventIn "
                 << (*fieldValue) << endl;
            return;
        }

        if (b->get()) // set_bind TRUE
        {
            if (this != current)
            {
                if (current)
                    current->eventOut(timeStamp, "isBound", VrmlSFBool(false));
                d_scene->bindablePush(this);
                eventOut(timeStamp, "isBound", VrmlSFBool(true));
            }
        }
        else // set_bind FALSE
        {
            d_scene->bindableRemove(this);
            if (this == current)
            {
                eventOut(timeStamp, "isBound", VrmlSFBool(false));
                current = d_scene->bindableNavigationInfoTop();
                if (current)
                    current->eventOut(timeStamp, "isBound", VrmlSFBool(true));
            }
        }
    }

    else
    {
        VrmlNode::eventIn(timeStamp, eventName, fieldValue);
    }
}

// Set the value of one of the node fields.

void VrmlNodeNavigationInfo::setField(const char *fieldName,
                                      const VrmlField &fieldValue)
{
    if (strcmp(fieldName, "scale") == 0)
    {
        if (fieldValue.toSFFloat())
            d_lastScale = (VrmlSFFloat &)fieldValue;
    }
    if
        TRY_FIELD(avatarSize, MFFloat)
    else if
        TRY_FIELD(headlight, SFBool)
    else if
        TRY_FIELD(speed, SFFloat)
    else if
        TRY_FIELD(scale, SFFloat)
    else if
        TRY_FIELD(near, SFFloat)
    else if
        TRY_FIELD(far, SFFloat)
    else if
        TRY_FIELD(type, MFString)
    else if
        TRY_FIELD(transitionType, MFString)
    else if
        TRY_FIELD(transitionTime, MFFloat)
    else if
        TRY_FIELD(visibilityLimit, SFFloat)
    else
        VrmlNodeChild::setField(fieldName, fieldValue);
}

const VrmlField *VrmlNodeNavigationInfo::getField(const char *fieldName) const
{
    if (strcmp(fieldName, "avatarSize") == 0)
        return &d_avatarSize;
    else if (strcmp(fieldName, "headlight") == 0)
        return &d_headlight;
    else if (strcmp(fieldName, "speed") == 0)
        return &d_speed;
    else if (strcmp(fieldName, "scale") == 0)
        return &d_scale;
    else if (strcmp(fieldName, "near") == 0)
        return &d_near;
    else if (strcmp(fieldName, "far") == 0)
        return &d_far;
    else if (strcmp(fieldName, "type") == 0)
        return &d_type;
    else if (strcmp(fieldName, "transitionType") == 0)
        return &d_transitionType;
    else if (strcmp(fieldName, "transitionTime") == 0)
        return &d_transitionTime;
    else if (strcmp(fieldName, "visibilityLimit") == 0)
        return &d_visibilityLimit;

    return VrmlNodeChild::getField(fieldName);
}
