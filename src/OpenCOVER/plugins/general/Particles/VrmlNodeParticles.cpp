/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 2001 Uwe Woessner
//
//  %W% %G%
//  VrmlNodeParticles.cpp

#include "VrmlNodeParticles.h"

#include "ParticleViewer.h"

static VrmlNode *creator(VrmlScene *scene)
{
    return new VrmlNodeParticles(scene);
}

VrmlNodeParticles::VrmlNodeParticles(VrmlScene *scene)
    : VrmlNodeChild(scene, typeName())
{
}

VrmlNodeParticles::VrmlNodeParticles(const VrmlNodeParticles &n)
    : VrmlNodeChild(n)
{
}


void VrmlNodeParticles::initFields(VrmlNodeParticles *node, VrmlNodeType *t)
{
    VrmlNodeChild::initFields(node, t); // Parent class
    initFieldsHelper(node, t,
        exposedField("particlesColorMap", node->d_ParticlesColorMap, [node](auto f)
            { ParticleViewer::instance()->setParticlesColorMap(node->d_ParticlesColorMap.get()); }),
        exposedField("particlesValue", node->d_ParticlesValue, [node](auto f)
            { ParticleViewer::instance()->setParticlesValue(node->d_ParticlesValue.get()); }),
        exposedField("particlesMin", node->d_ParticlesMin, [node](auto f)
            { ParticleViewer::instance()->setParticlesMin(node->d_ParticlesMin.get()); }),
        exposedField("particlesMax", node->d_ParticlesMax, [node](auto f)
            { ParticleViewer::instance()->setParticlesMax(node->d_ParticlesMax.get()); }),
        exposedField("particlesRadius", node->d_ParticlesRadius, [node](auto f)
            { ParticleViewer::instance()->setParticlesRadius(node->d_ParticlesRadius.get()); }),
        exposedField("particlesRadiusValue", node->d_ParticlesRadiusValue, [node](auto f)
            { ParticleViewer::instance()->setParticlesRadiusValue(node->d_ParticlesRadiusValue.get()); }),
        exposedField("arrowsColorMap", node->d_ArrowsColorMap, [node](auto f)
            { ParticleViewer::instance()->setArrowsColorMap(node->d_ArrowsColorMap.get()); }),
        exposedField("arrowsValue", node->d_ArrowsValue, [node](auto f)
            { ParticleViewer::instance()->setArrowsValue(node->d_ArrowsValue.get()); }),
        exposedField("arrowsMin", node->d_ArrowsMin, [node](auto f)
            { ParticleViewer::instance()->setArrowsMin(node->d_ArrowsMin.get()); }),
        exposedField("arrowsMax", node->d_ArrowsMax, [node](auto f)
            { ParticleViewer::instance()->setArrowsMax(node->d_ArrowsMax.get()); }),
        exposedField("arrowsRadius", node->d_ArrowsRadius, [node](auto f)
            { ParticleViewer::instance()->setArrowsRadius(node->d_ArrowsRadius.get()); }),
        exposedField("arrowsRadiusValue", node->d_ArrowsRadiusValue, [node](auto f)
            { ParticleViewer::instance()->setArrowsRadiusValue(node->d_ArrowsRadiusValue.get()); }));
    if (t)
    {
        t->addEventIn("update", VrmlField::SFTIME);
    }
}
const char *VrmlNodeParticles::typeName()
{
    return "Particles";
}

inline void VrmlNodeParticles::eventIn(double timeStamp, const char *eventName, const VrmlField *fieldValue)
{
    if (strcmp(eventName, "update") == 0)
    {
        ParticleViewer::instance()->updateColors();
    }
    // Check parent class for eventIn
    else
    {
        VrmlNode::eventIn(timeStamp, eventName, fieldValue);
    }
}
