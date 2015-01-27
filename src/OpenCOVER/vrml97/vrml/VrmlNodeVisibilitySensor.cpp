/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeVisibilitySensor.cpp

#include "config.h"
#include "VrmlNodeVisibilitySensor.h"
#include "VrmlNodeType.h"
#include "VrmlNodeNavigationInfo.h"

#include "MathUtils.h"
#include "System.h"
#include "Viewer.h"
#include "VrmlScene.h"

using namespace vrml;

// VisibilitySensor factory.

static VrmlNode *creator(VrmlScene *scene)
{
    return new VrmlNodeVisibilitySensor(scene);
}

// Define the built in VrmlNodeType:: "VisibilitySensor" fields

VrmlNodeType *VrmlNodeVisibilitySensor::defineType(VrmlNodeType *t)
{
    static VrmlNodeType *st = 0;

    if (!t)
    {
        if (st)
            return st; // Only define the type once.
        t = st = new VrmlNodeType("VisibilitySensor", creator);
    }

    VrmlNodeChild::defineType(t); // Parent class
    t->addExposedField("center", VrmlField::SFVEC3F);
    t->addExposedField("enabled", VrmlField::SFBOOL);
    t->addExposedField("size", VrmlField::SFVEC3F);
    t->addEventOut("enterTime", VrmlField::SFTIME);
    t->addEventOut("exitTime", VrmlField::SFTIME);
    t->addEventOut("isActive", VrmlField::SFBOOL);

    return t;
}

VrmlNodeType *VrmlNodeVisibilitySensor::nodeType() const { return defineType(0); }

VrmlNodeVisibilitySensor::VrmlNodeVisibilitySensor(VrmlScene *scene)
    : VrmlNodeChild(scene)
    , d_center(0.0, 0.0, 0.0)
    , d_enabled(true)
    , d_size(0.0, 0.0, 0.0)
    , d_isActive(false)
    , d_enterTime(0.0)
    , d_exitTime(0.0)
{
    setModified();
}

VrmlNodeVisibilitySensor::~VrmlNodeVisibilitySensor()
{
}

VrmlNode *VrmlNodeVisibilitySensor::cloneMe() const
{
    return new VrmlNodeVisibilitySensor(*this);
}

std::ostream &VrmlNodeVisibilitySensor::printFields(std::ostream &os, int indent)
{
    if (!FPZERO(d_center.x()) || !FPZERO(d_center.y()) || !FPZERO(d_center.z()))
        PRINT_FIELD(center);
    if (!d_enabled.get())
        PRINT_FIELD(enabled);
    if (!FPZERO(d_size.x()) || !FPZERO(d_size.y()) || !FPZERO(d_size.z()))
        PRINT_FIELD(size);

    return os;
}

//
// Generate visibility events.
//
// This is in a render() method since the it needs to be computed
// with respect to the accumulated transformations above it in the
// scene graph. Move to update() when xforms are accumulated in Groups...
//

void VrmlNodeVisibilitySensor::render(Viewer *viewer)
{

    if (d_enabled.get())
    {
        VrmlSFTime timeNow(System::the->time());
        float xyz[2][3];

        // hack: enclose box in a sphere...
        xyz[0][0] = d_center.x();
        xyz[0][1] = d_center.y();
        xyz[0][2] = d_center.z();
        xyz[1][0] = d_center.x() + d_size.x();
        xyz[1][1] = d_center.y() + d_size.y();
        xyz[1][2] = d_center.z() + d_size.z();
        viewer->transformPoints(2, &xyz[0][0]);
        float dx = xyz[1][0] - xyz[0][0];
        float dy = xyz[1][1] - xyz[0][1];
        float dz = xyz[1][2] - xyz[0][2];
        float r = dx * dx + dy * dy + dz * dz;
        if (!FPZERO(r))
            r = sqrt(r);

        // Was the sphere visible last time through? How does this work
        // for USE'd nodes? I need a way for each USE to store whether
        // it was active.
        bool wasIn = d_isActive.get();

        // Is the sphere visible? ...
        bool inside = xyz[0][2] < 0.0; // && z > - scene->visLimit()
        if (inside)
        {
            VrmlNodeNavigationInfo *ni = d_scene->bindableNavigationInfoTop();
            if (ni && !FPZERO(ni->visibilityLimit()) && xyz[0][2] < -ni->visibilityLimit())
                inside = false;
        }

        // This bit assumes 90degree fieldOfView to get rid of trig calls...
        if (inside)
            inside = (fabs(xyz[0][0]) < -0.5 * xyz[0][2] + r && fabs(xyz[0][1]) < -0.5 * xyz[0][2] + r);

        // Just became visible
        if (inside && !wasIn)
        {
            System::the->debug("VS enter %g, %g, %g\n",
                               xyz[0][0], xyz[0][1], xyz[0][2]);

            d_isActive.set(true);
            eventOut(timeNow.get(), "isActive", d_isActive);

            d_enterTime = timeNow;
            eventOut(timeNow.get(), "enterTime", d_enterTime);
        }

        // Check if viewer has left the box
        else if (wasIn && !inside)
        {
            System::the->debug("VS exit %g, %g, %g\n",
                               xyz[0][0], xyz[0][1], xyz[0][2]);

            d_isActive.set(false);
            eventOut(timeNow.get(), "isActive", d_isActive);

            d_exitTime = timeNow;
            eventOut(timeNow.get(), "exitTime", d_exitTime);
        }
    }

    else
        clearModified();
}

// Set the value of one of the node fields.

void VrmlNodeVisibilitySensor::setField(const char *fieldName,
                                        const VrmlField &fieldValue)
{
    if
        TRY_FIELD(center, SFVec3f)
    else if
        TRY_FIELD(enabled, SFBool)
    else if
        TRY_FIELD(size, SFVec3f)
    else
        VrmlNodeChild::setField(fieldName, fieldValue);
}

const VrmlField *VrmlNodeVisibilitySensor::getField(const char *fieldName) const
{
    if (strcmp(fieldName, "center") == 0)
        return &d_center;
    else if (strcmp(fieldName, "enabled") == 0)
        return &d_enabled;
    else if (strcmp(fieldName, "size") == 0)
        return &d_size;

    return VrmlNodeChild::getField(fieldName);
}
