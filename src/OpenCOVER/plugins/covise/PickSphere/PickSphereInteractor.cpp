/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <cover/coVRPluginSupport.h>
#include <cover/coVRMSController.h>
#include <cover/VRSceneGraph.h>
#include <cover/coInteractor.h>
#include "PickSphereInteractor.h"
#include "PickSpherePlugin.h"

#include <math.h>

#include <osg/MatrixTransform>
#include <osg/PositionAttitudeTransform>
#include <osg/Geode>
#include <osg/Geometry>
#include <osg/Material>
#include <osg/ShapeDrawable>
#include <osg/LineWidth>
#include <osg/PolygonMode>

#include <config/CoviseConfig.h>

#include <cover/coVRAnimationManager.h>
#include <cover/coVRMSController.h>

using namespace covise;
using namespace opencover;
using namespace osg;

PickSphereInteractor::PickSphereInteractor(coInteraction::InteractionType type, const char *name, coInteraction::InteractionPriority priority = Medium)
    : coTrackerButtonInteraction(type, name, priority)
    , m_selectedWithBox(false)
{
    m_spheres = NULL;
    m_lastIndex = -1;
    m_lastIndexTime = -1.0;

    highlight = new MatrixTransform;
    osg::StateSet *state = highlight->getOrCreateStateSet();

    osg::PolygonMode *pm = new osg::PolygonMode;
    pm->setMode(osg::PolygonMode::FRONT_AND_BACK, osg::PolygonMode::LINE);
    state->setAttributeAndModes(pm, osg::StateAttribute::ON);

    osg::LineWidth *lw = new osg::LineWidth(2.);
    state->setAttributeAndModes(lw, osg::StateAttribute::ON);

    osg::Vec4 magenta(1., 0., 1., 1.);

    osg::Geode *geode = new osg::Geode;
    highlight->addChild(geode);

    osg::ShapeDrawable *shape = new osg::ShapeDrawable(new osg::Sphere);
    osg::TessellationHints *tess = new osg::TessellationHints;
    tess->setDetailRatio(0.3);
    shape->setTessellationHints(tess);
    shape->setColor(magenta);
    geode->addDrawable(shape);

    osg::Geometry *cross = new osg::Geometry();
    osg::Vec3Array *vert = new osg::Vec3Array;
    osg::DrawArrayLengths *primitives = new osg::DrawArrayLengths(osg::PrimitiveSet::LINES);
    primitives->push_back(6);
    vert->push_back(osg::Vec3(-5., 0., 0.));
    vert->push_back(osg::Vec3(5., 0., 0.));
    vert->push_back(osg::Vec3(0., -5., 0.));
    vert->push_back(osg::Vec3(0., 5., 0.));
    vert->push_back(osg::Vec3(0., 0., -5.));
    vert->push_back(osg::Vec3(0., 0., 5.));
    cross->setVertexArray(vert);
    cross->addPrimitiveSet(primitives);
    osg::Vec4Array *colors = new osg::Vec4Array();
    colors->push_back(magenta);
    cross->setColorArray(colors);
    cross->setColorBinding(osg::Geometry::BIND_OVERALL);
    geode->addDrawable(cross);

    highlight->ref();
}

PickSphereInteractor::~PickSphereInteractor()
{
    highlightSphere(-1);
    highlight->unref();
}

void
PickSphereInteractor::startInteraction()
{
    if (cover->debugLevel(3))
        fprintf(stderr, "\nPickSphereInteractor::startMove\n");

    m_lastIndex = -1;

    // store hand mat
    Matrix initHandMat = cover->getPointerMat();
    // get postion and direction of pointer
    m_initHandPos = initHandMat.preMult(Vec3(0.0, 0.0, 0.0));
    m_initHandDirection = initHandMat.preMult(Vec3(0.0, 1.0, 0.0));
    //stop animation
    m_selectionChanged = false;
    m_animationWasRunning = coVRAnimationManager::instance()->animationRunning();
    if (m_animationWasRunning)
        enableAnimation(false);
}

void
PickSphereInteractor::enableAnimation(bool state)
{
    coVRAnimationManager::instance()->enableAnimation(state);
}

void
PickSphereInteractor::stopInteraction()
{
    if (cover->debugLevel(3))
        fprintf(stderr, "\nPickSphereInteractor::stopMove\n");

    highlightSphere(-1);

    if (coVRMSController::instance()->isMaster())
    {
        int sphereIndex = hitSphere();
        if (sphereIndex == -1)
        {
            if (cover->frameTime() - m_lastIndexTime < 1.0)
                sphereIndex = m_lastIndex;
        }

        if (cover->debugLevel(3))
            fprintf(stderr, "\nPickSphereInteractor::stopInteraction\n");

        if (sphereIndex != -1)
            addSelectedSphere(sphereIndex);
    }

    if (m_animationWasRunning)
        enableAnimation(true);
}

int PickSphereInteractor::hitSphere()
{
    if (!m_spheres)
    {
        printf("renderer received no sphere representation to pick.\nTry to execute again to fix this problem\n");
        return -1;
    }
    int currentFrame = coVRAnimationManager::instance()->getAnimationFrame();
    if (currentFrame < 0 || currentFrame > (*m_spheres).size())
        return -1;
    const SphereData actualSpheres = *(*m_spheres)[currentFrame];

    Matrix currHandMat = cover->getPointerMat();
    Matrix invBase = cover->getInvBaseMat();

    currHandMat = currHandMat * invBase;

    Vec3 currHandBegin = currHandMat.preMult(Vec3(0.0, 0.0, 0.0));
    Vec3 currHandEnd = currHandMat.preMult(Vec3(0.0, 1.0, 0.0));
    Vec3 currHandDirection = currHandEnd - currHandBegin;

    float nearestDistance = FLT_MAX;
    int sphereIndex = -1;

    for (int i = 0; i < actualSpheres.n; i++)
    {
        float x, y, z, radius;
        double distance;
        x = actualSpheres.x[i];
        y = actualSpheres.y[i];
        z = actualSpheres.z[i];
        radius = actualSpheres.r[i];
        const float correct = PickSpherePlugin::getScale();
        distance = LineSphereIntersect(Vec3(x, y, z), radius * correct, currHandBegin, currHandDirection);
        if (distance > 0 && distance < nearestDistance)
        {
            nearestDistance = distance;
            sphereIndex = i;
        }
    }

    return sphereIndex;
}

bool
PickSphereInteractor::getMultipleSelect()
{
    return m_multipleSelect;
}

bool
PickSphereInteractor::selectionChanged()
{
    if (coVRMSController::instance()->isMaster())
    {
        coVRMSController::instance()->sendSlaves(&m_selectionChanged, sizeof(m_selectionChanged));
    }
    else
    {
        coVRMSController::instance()->readMaster(&m_selectionChanged, sizeof(m_selectionChanged));
    }
    if (m_selectionChanged)
    {
        return true;
    }
    else
        return false;
}

void
PickSphereInteractor::doInteraction()
{
    if (cover->debugLevel(4))
        fprintf(stderr, "\nPickSphereInteractor::doInteraction\n");

    int index = -1;
    if (coVRMSController::instance()->isMaster())
    {
        index = hitSphere();
        if (index != -1)
        {
            m_lastIndex = index;
            m_lastIndexTime = cover->frameTime();
        }
        coVRMSController::instance()->sendSlaves(&index, sizeof(index));
    }
    else
    {
        coVRMSController::instance()->readMaster(&index, sizeof(index));
        if (index != -1)
        {
            m_lastIndex = index;
            m_lastIndexTime = cover->frameTime();
        }
    }

    highlightSphere(index);

    if (cover->debugLevel(4))
        fprintf(stderr, "\nPickSphereInteractor::doInteraction\n");
}

void
PickSphereInteractor::highlightSphere(int index)
{
    if (index == -1)
    {
        while (highlight->getNumParents() > 0)
            highlight->getParent(0)->removeChild(highlight.get());
    }
    else
    {
        float x[4];
        if (coVRMSController::instance()->isMaster())
        {
            int currentFrame = coVRAnimationManager::instance()->getAnimationFrame();
            if (currentFrame < 0 || currentFrame > (*m_spheres).size())
                return;
            const SphereData actualSpheres = *(*m_spheres)[currentFrame];

            x[0] = actualSpheres.x[index];
            x[1] = actualSpheres.y[index];
            x[2] = actualSpheres.z[index];
            x[3] = actualSpheres.r[index] * PickSpherePlugin::getScale() * 1.1;
            coVRMSController::instance()->sendSlaves(x, sizeof(x));
        }
        else
        {
            coVRMSController::instance()->readMaster(x, sizeof(x));
        }

        osg::Matrix mat;
        mat.makeScale(x[3], x[3], x[3]);
        mat.setTrans(osg::Vec3(x[0], x[1], x[2]));
        highlight->setMatrix(mat);

        VRSceneGraph::instance()->objectsRoot()->addChild(highlight.get());
    }
}

void
PickSphereInteractor::boxSelect(Vec3 min, Vec3 max)
{
    setSelectedWithBox(true);
    if (coVRMSController::instance()->isSlave())
        return;

    if (!m_spheres)
    {
        printf("renderer received no sphere representation to pick.\nTry to execute again to fix this problem\n");
        return;
    }

    enableMultipleSelect(true);
    int currentFrame = coVRAnimationManager::instance()->getAnimationFrame();
    if (currentFrame < 0 || currentFrame > (*m_spheres).size())
        return;
    const SphereData actualSpheres = *(*m_spheres)[currentFrame];

    if (min.x() > max.x())
        std::swap(min.x(), max.x());
    if (min.y() > max.y())
        std::swap(min.y(), max.y());
    if (min.z() > max.z())
        std::swap(min.z(), max.z());

    printf("in boxSelect\n");
    osg::BoundingBox box(min, max);
    for (int i = 0; i < actualSpheres.n; i++)
    {
        const Vec3 *center = new Vec3(actualSpheres.x[i], actualSpheres.y[i], actualSpheres.z[i]);
        if (box.contains(*center))
        {
            printf("sphere %d in box\n", i);
            addSelectedSphere(i);
        }
    }
}

void
PickSphereInteractor::enableMultipleSelect(bool multipleSelect)
{
    this->m_multipleSelect = multipleSelect;
}

void
PickSphereInteractor::addSelectedSphere(int sphereIndex)
{
    //multipleSelect
    if (!m_multipleSelect)
    {
        m_selectionChanged = true;
        m_evaluateIndices.clear();
    }

    if (sphereIndex >= 0)
    {
        m_selectionChanged = true;
        m_evaluateIndices.add(sphereIndex);
    }
    else
    {
        return;
    }
}

std::string
PickSphereInteractor::getSelectedParticleString()
{
    return m_evaluateIndices.getRestraintString();
}

int
PickSphereInteractor::getSelectedParticleCount()
{
    return m_evaluateIndices.getValues().size();
}

void
PickSphereInteractor::updateSelection(const char *particles)
{
    m_evaluateIndices.clear();
    m_evaluateIndices.add(particles);
}

double
PickSphereInteractor::LineSphereIntersect(Vec3 center, float radius, Vec3 handPos, Vec3 handDirection)
{
    handDirection.normalize();
    Vec3 pMinusC = handPos - center;
    double b = handDirection * pMinusC;
    double c = pMinusC * pMinusC - radius * radius;
    double discrm = b * b - c;

    if (discrm < 0.)
        return -1.;

    double d = sqrt(discrm);
    double t = -b - d;
    if (t >= 0.)
        return t;
    t = -b + d;
    if (t >= 0.)
        return t;

    return -1.;
}
