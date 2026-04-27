/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _JSBSIM_VRML_NODE_THERMAL_H
#define _JSBSIM_VRML_NODE_THERMAL_H

#include <util/common.h>

#include <vrml97/vrml/VrmlNodeChild.h>
#include <vrml97/vrml/VrmlSFFloat.h>
#include <vrml97/vrml/VrmlSFVec3f.h>

using namespace vrml;

class PLUGINEXPORT VrmlNodeThermal : public VrmlNodeChild
{
public:
    static void initFields(VrmlNodeThermal *node, VrmlNodeType *t);
    static const char *typeName();

    VrmlNodeThermal(VrmlScene *scene = 0);
    VrmlNodeThermal(const VrmlNodeThermal &n);
    virtual ~VrmlNodeThermal();

    void eventIn(double timeStamp, const char *eventName,
        const VrmlField *fieldValue);

    virtual void render(Viewer *);

    VrmlSFVec3f d_direction;
    VrmlSFVec3f d_location;
    VrmlSFFloat d_maxBack;
    VrmlSFFloat d_maxFront;
    VrmlSFFloat d_minBack;
    VrmlSFFloat d_minFront;
    VrmlSFFloat d_height;
    VrmlSFVec3f d_velocity;
    VrmlSFFloat d_turbulence;

private:
};

#endif
