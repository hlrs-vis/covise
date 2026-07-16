/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */
#include "VrmlNodeViewPoint.h"
#include <cover/ui/Slider.h>
#include <vrml97/vrml/VrmlSFFloat.h>

#include "ViewPoint.h"

static VrmlNode *creator(VrmlScene *scene)
{
    return new VrmlNodeViewPoint(scene);
}

VrmlNodeViewPoint::VrmlNodeViewPoint(VrmlScene *scene)
    : VrmlNodeChild(scene, typeName())
{
}

VrmlNodeViewPoint::VrmlNodeViewPoint(const VrmlNodeViewPoint &n)
    : VrmlNodeChild(n)
{
}

void VrmlNodeViewPoint::initFields(VrmlNodeViewPoint *node, VrmlNodeType *t)
{
    VrmlNodeChild::initFields(node, t); // Parent class
    initFieldsHelper(node, t,
        exposedField("transitionDuration", node->d_transitionDuration, [node](auto f)
            { ViewPoints::instance()->speedSlider->setValue(node->d_transitionDuration.get());  ViewPoints::instance()->flightTime = node->d_transitionDuration.get(); }),
        exposedField("viewPointName", node->d_viewPointName, [node](auto n)
            { ViewPoints::instance()->setName(node->d_viewPointName.get()); }));
}

const char *VrmlNodeViewPoint::typeName()
{
    return "ViewPoint";
}
