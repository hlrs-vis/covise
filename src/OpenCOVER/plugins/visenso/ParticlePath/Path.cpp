/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "Path.h"
#include "Const.h"

#include <cover/VRSceneGraph.h>
#include <cover/coVRPluginSupport.h>
using namespace opencover;
using namespace covise;

Path::Path(osg::ref_ptr<osg::Group> parent)
    : parentNode(parent)
    , animationTime(-ANIMATION_PAUSE)
{
    tracer = new Tracer();

    pathGeode = new osg::Geode();
    pathGeometry = new osg::Geometry();
    pathGeode->addDrawable(pathGeometry.get());
    parentNode->addChild(pathGeode.get());

    particle = new Particle(parentNode);
    velocityArrow = new Arrow(parentNode, VELOCITY_ARROW_COLOR);
    combinedForceArrow = new Arrow(parentNode, COMBINED_FORCE_ARROW_COLOR);
    electricForceArrow = new Arrow(parentNode, ELECTRIC_FORCE_ARROW_COLOR);
    magneticForceArrow = new Arrow(parentNode, MAGNETIC_FORCE_ARROW_COLOR);
}

Path::~Path()
{
    parentNode->removeChild(pathGeode.get());
    delete particle;
    delete velocityArrow;
    delete combinedForceArrow;
    delete electricForceArrow;
    delete magneticForceArrow;
    delete tracer;
}

void Path::preFrame()
{
    if (tracer->result.size() == 0)
        return;

    animationTime += cover->frameDuration() * ANIMATION_SPEED;
    if (animationTime >= (float)tracer->result.size())
        animationTime = 0.0f;

    int timeIndex;
    if (animationTime < 0.0f)
    {
        timeIndex = 0;
    }
    else
    {
        timeIndex = int(animationTime);
    }

    particle->update(tracer->result[timeIndex].position);
    velocityArrow->update(tracer->result[timeIndex].position, tracer->result[timeIndex].velocity * VELOCITY_ARROW_SCALING);
    combinedForceArrow->update(tracer->result[timeIndex].position, tracer->result[timeIndex].combinedForce * FORCE_ARROW_SCALING);
    electricForceArrow->update(tracer->result[timeIndex].position, tracer->result[timeIndex].electricForce * FORCE_ARROW_SCALING);
    magneticForceArrow->update(tracer->result[timeIndex].position, tracer->result[timeIndex].magneticForce * FORCE_ARROW_SCALING);
}

void Path::calculateNewPath()
{
    tracer->trace();

    animationTime = -ANIMATION_PAUSE;

    combinedForceArrow->setVisible((tracer->config.electricField != 0.0) && (tracer->config.magneticField != 0.0));

    if (tracer->config.charge == 0.0)
        particle->setColor(PARTICLE_COLOR);
    else if (tracer->config.charge > 0.0)
        particle->setColor(PARTICLE_COLOR_POSITIVE);
    else
        particle->setColor(PARTICLE_COLOR_NEGATIVE);

    // update path geometry

    for (int i = 0; i < pathGeometry->getNumPrimitiveSets(); i++)
        pathGeometry->removePrimitiveSet(i);

    pathGeometry->setStateSet(VRSceneGraph::instance()->loadDefaultGeostate());
    pathGeometry->getOrCreateStateSet()->setAttributeAndModes(new osg::LineWidth(PATH_WIDTH), osg::StateAttribute::ON);
    osg::ref_ptr<osg::Vec4Array> pathColor = new osg::Vec4Array();
    pathColor->push_back(PATH_COLOR);
    pathGeometry->setColorArray(pathColor.get());
    pathGeometry->setColorBinding(osg::Geometry::BIND_OVERALL);
    pathGeometry->getStateSet()->setMode(GL_LIGHTING, osg::StateAttribute::OFF);

    osg::Vec3Array *vertices = new osg::Vec3Array;
    osg::DrawElementsUInt *conns = new osg::DrawElementsUInt(osg::PrimitiveSet::LINE_STRIP, 0);
    for (int i = 0; i < tracer->result.size(); ++i)
    {
        conns->push_back(i);
        vertices->push_back(tracer->result[i].position);
    }
    pathGeometry->addPrimitiveSet(conns);
    pathGeometry->setVertexArray(vertices);
}

void Path::setInactive()
{
    particle->setVisible(false);
    velocityArrow->setVisible(false);
    combinedForceArrow->setVisible(false);
    electricForceArrow->setVisible(false);
    magneticForceArrow->setVisible(false);

    osg::ref_ptr<osg::Vec4Array> pathColor = new osg::Vec4Array();
    pathColor->push_back(PREVIOUS_PATH_COLOR);
    pathGeometry->setColorArray(pathColor.get());
}
