/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _VRML_NODE_PARTICLES_H
#define _VRML_NODE_PARTICLES_H

#include <util/coTypes.h>

#include <vrml97/vrml/VrmlNode.h>
#include <vrml97/vrml/VrmlSFFloat.h>
#include <vrml97/vrml/VrmlSFVec3f.h>
#include <vrml97/vrml/VrmlSFString.h>
#include <vrml97/vrml/VrmlNodeChild.h>

using namespace vrml;

class VrmlNodeParticles : public VrmlNodeChild
{
public:
    VrmlNodeParticles(VrmlScene *scene = 0);
    VrmlNodeParticles(const VrmlNodeParticles &n);

    static void initFields(VrmlNodeParticles *node, vrml::VrmlNodeType *t);
    static const char *typeName();

    void eventIn(double timeStamp,
        const char *eventName,
        const VrmlField *fieldValue);

private:
    VrmlSFString d_ParticlesColorMap = "";
    VrmlSFString d_ParticlesValue = "";
    VrmlSFFloat d_ParticlesMin = 0.0;
    VrmlSFFloat d_ParticlesMax = 1.0;
    VrmlSFFloat d_ParticlesRadius = 1.0;
    VrmlSFString d_ParticlesRadiusValue = "";
    VrmlSFString d_ArrowsColorMap = "";
    VrmlSFString d_ArrowsValue = "";
    VrmlSFFloat d_ArrowsMin = 0.0;
    VrmlSFFloat d_ArrowsMax = 1.0;
    VrmlSFFloat d_ArrowsRadius = 1.0;
    VrmlSFString d_ArrowsRadiusValue = "";
};
#endif
