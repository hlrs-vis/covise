/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <cover/coVRPluginSupport.h>
#include <cover/VRSceneGraph.h>
#include <cover/coVRMSController.h>
#include "BoxSelectionInteractor.h"

#include <math.h>
#include <osg/MatrixTransform>
#include <osg/PositionAttitudeTransform>
#include <osg/Geode>
#include <osg/Geometry>
#include <osg/Material>
#include <osg/ShapeDrawable>
#include <osg/LineWidth>

#include <config/CoviseConfig.h>

#include <cover/coVRAnimationManager.h>

using namespace osg;
using namespace covise;
using namespace opencover;

BoxSelectionInteractor::BoxSelectionInteractor(
    coInteraction::InteractionType type, const char *name, coInteraction::InteractionPriority priority = Medium)
    : coTrackerButtonInteraction(type, name, priority)
    , m_min(osg::Vec3(0, 0, 0))
    , m_max(osg::Vec3(0, 0, 0))
    , m_worldCoordMin(osg::Vec3(0, 0, 0))
    , m_worldCoordMax(osg::Vec3(0, 0, 0))
    , m_cubeGeode(NULL)
    , m_root(NULL)
    , m_interactionFinished(NULL)
    , m_interactionRunning(NULL)
{
}

BoxSelectionInteractor::~BoxSelectionInteractor()
{
}

void
BoxSelectionInteractor::startInteraction()
{
    if (cover->debugLevel(3))
        fprintf(stderr, "\nBoxSelectionInteractor::startMove\n");

    Matrix invBase = cover->getInvBaseMat();
    m_worldCoordMin = VRSceneGraph::instance()->getWorldPointOfInterest();
    m_min = invBase.preMult(m_worldCoordMin);

    //stop animation
    m_animationWasRunning = coVRAnimationManager::instance()->animationRunning();
    if (m_animationWasRunning)
        coVRAnimationManager::instance()->enableAnimation(false);

    // create nodes
    m_root = new Group();
    // geometry node
    m_cubeGeode = new Geode();
    // stateset containing the material and line width
    StateSet *cubeState = createBoxMaterial();
    // set state set of the cube geode
    m_cubeGeode->setStateSet(cubeState);

    // create scenegraph
    cover->getScene()->addChild(m_root.get());
    m_root->addChild(m_cubeGeode);

    updateBox(m_worldCoordMin, m_worldCoordMin);
}

void
BoxSelectionInteractor::doInteraction()
{
    Matrix invBase = cover->getInvBaseMat();

    m_worldCoordMax = VRSceneGraph::instance()->getWorldPointOfInterest();
    m_max = invBase.preMult(m_worldCoordMax);

    updateBox(m_worldCoordMin, m_worldCoordMax);

    if (m_interactionRunning)
        m_interactionRunning();

    if (cover->debugLevel(3))
        fprintf(stderr, "\nBoxSelectionInteractor::move\n");
}

void
BoxSelectionInteractor::stopInteraction()
{
    if (cover->debugLevel(3))
        fprintf(stderr, "\nBoxSelectionInteractor::stopMove\n");

    Matrix invBase = cover->getInvBaseMat();

    m_worldCoordMax = VRSceneGraph::instance()->getWorldPointOfInterest();
    m_max = invBase.preMult(m_worldCoordMax);

    if (m_interactionFinished)
        m_interactionFinished();

    updateBox(m_worldCoordMin, m_worldCoordMax);

    // delete nodes and scenegraph
    if (m_cubeGeode)
    {
        if (m_cubeGeode->getNumParents())
        {
            m_cubeGeode->getParent(0)->removeChild(m_cubeGeode);
        }
    }
    coVRAnimationManager::instance()->enableAnimation(m_animationWasRunning);
}

void BoxSelectionInteractor::registerInteractionRunningCallback(void (*interactionRunning)())
{
    m_interactionRunning = interactionRunning;
}

void BoxSelectionInteractor::unregisterInteractionRunningCallback()
{
    m_interactionRunning = NULL;
}

void BoxSelectionInteractor::registerInteractionFinishedCallback(void (*interactionFinished)())
{
    m_interactionFinished = interactionFinished;
}

void BoxSelectionInteractor::unregisterInteractionFinishedCallback()
{
    m_interactionFinished = NULL;
}

void BoxSelectionInteractor::getBox(float &minX, float &minY, float &minZ, float &maxX, float &maxY, float &maxZ)
{
    minX = m_min.x();
    minY = m_min.y();
    minZ = m_min.z();
    maxX = m_max.x();
    maxY = m_max.y();
    maxZ = m_max.z();
}

Geometry *
BoxSelectionInteractor::createWireframeBox(Vec3 min, Vec3 max)
{
    //printf("%f, %f \n", min.x(), max.x());
    //printf("%f, %f \n", min.y(), max.y());
    //printf("%f, %f \n", min.z(), max.z());
    if (cover->debugLevel(3))
        fprintf(stderr, "\nCubeInteractor::createWireframeUnitCube\n");

    Geometry *cube = new Geometry();

    // create the coordinates of the cube

    //      5.......6
    //    .       . .
    //  4.......7   .
    //  .       .   .
    //  .   1   .   2
    //  .       . .
    //  0.......3

    Vec3Array *cubeVertices = new Vec3Array;
    cubeVertices->push_back(Vec3(min.x(), min.y(), min.z()));
    cubeVertices->push_back(Vec3(min.x(), max.y(), min.z()));
    cubeVertices->push_back(Vec3(max.x(), max.y(), min.z()));
    cubeVertices->push_back(Vec3(max.x(), min.y(), min.z()));

    cubeVertices->push_back(Vec3(min.x(), min.y(), max.z()));
    cubeVertices->push_back(Vec3(min.x(), max.y(), max.z()));
    cubeVertices->push_back(Vec3(max.x(), max.y(), max.z()));
    cubeVertices->push_back(Vec3(max.x(), min.y(), max.z()));

    // add the coordinates to the geometry object
    cube->setVertexArray(cubeVertices);

    // create the lines
    DrawElementsUInt *wireframe = new DrawElementsUInt(PrimitiveSet::LINES, 0);
    // line 0-3
    wireframe->push_back(0);
    wireframe->push_back(3);
    // line 3-7
    wireframe->push_back(3);
    wireframe->push_back(7);
    // line 7-4
    wireframe->push_back(7);
    wireframe->push_back(4);
    // line 4-0
    wireframe->push_back(4);
    wireframe->push_back(0);
    // line 1-2
    wireframe->push_back(1);
    wireframe->push_back(2);
    // line 2-6
    wireframe->push_back(2);
    wireframe->push_back(6);

    // line 6-5
    wireframe->push_back(6);
    wireframe->push_back(5);
    // line 5-1
    wireframe->push_back(5);
    wireframe->push_back(1);
    // line 4-5
    wireframe->push_back(4);
    wireframe->push_back(5);
    // line 7-6
    wireframe->push_back(7);
    wireframe->push_back(6);
    // line 0-1
    wireframe->push_back(0);
    wireframe->push_back(1);
    // line 3-2
    wireframe->push_back(3);
    wireframe->push_back(2);

    // add the lines to the geometry
    cube->addPrimitiveSet(wireframe);
    // create the green wireframe color
    Vec4Array *color = new Vec4Array;
    color->push_back(Vec4(0.0, 0.75, 0.0, 1.0));
    // add color to geometry
    cube->setColorArray(color);
    // specifying that color should be added to the whole cube
    cube->setColorBinding(Geometry::BIND_PER_PRIMITIVE_SET);
    return cube;
}

StateSet *
BoxSelectionInteractor::createBoxMaterial()
{
    if (cover->debugLevel(3))
        fprintf(stderr, "\nCubeInteractor::createWireframeCubeMaterial\n");

    // creating the material properties
    Material *mtl = new Material;
    mtl->setColorMode(Material::AMBIENT_AND_DIFFUSE);
    mtl->setAmbient(Material::FRONT_AND_BACK, Vec4(0.2f, 0.2f, 0.2f, 1.0f));
    mtl->setDiffuse(Material::FRONT_AND_BACK, Vec4(0.9f, 0.9f, 0.9f, 1.0f));
    mtl->setSpecular(Material::FRONT_AND_BACK, Vec4(0.9f, 0.9f, 0.9f, 1.0f));
    mtl->setEmission(Material::FRONT_AND_BACK, Vec4(0.0f, 0.0f, 0.0f, 1.0f));
    mtl->setShininess(Material::FRONT_AND_BACK, 16.0f);

    //create a new StateSet
    StateSet *set = new StateSet();
    //modify the attributes of the Geode-container
    //add color
    set->setAttributeAndModes(mtl, StateAttribute::ON);
    //define line width
    set->setAttribute(new LineWidth(3.5), StateAttribute::ON);

    return set;
}

void
BoxSelectionInteractor::updateBox(Vec3 min, Vec3 max)
{
    while (m_cubeGeode->getNumDrawables())
        m_cubeGeode->removeDrawable(m_cubeGeode->getDrawable(0));
    // adding cube
    m_cubeGeode->addDrawable(createWireframeBox(min, max));
}
