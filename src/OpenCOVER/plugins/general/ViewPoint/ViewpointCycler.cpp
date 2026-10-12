/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "ViewpointCycler.h"
#include <cover/ui/Owner.h>
#include <cover/coVRPluginList.h>

#include "plugins/general/ViewPoint/ViewPoint.h"

inline auto getViewpoints()
{
    return ViewPoints::instance()->getViewpoints();
}

ViewpointCycler::Effect::Effect(osg::Matrix initial)
    : m_initial(initial)
{
}
osg::Matrix ViewpointCycler::Effect::update(float progress)
{
    return m_initial;
}

ViewpointCycler::DollyEffect::DollyEffect(osg::Matrix initial, double distance)
    : ViewpointCycler::Effect(initial)
    , m_distance(distance)
{
}

osg::Matrix ViewpointCycler::DollyEffect::update(float progress)
{
    double distance = progress * m_distance;
    osg::Vec3 forward(0, 1, 0);
    return osg::Matrix::translate(forward * distance) * m_initial;
}

ViewpointCycler::OrbitEffect ::OrbitEffect(osg::Matrix initial, osg::Vec3 center, double angle)
    : ViewpointCycler::Effect(initial)
    , m_center(center)
    , m_angle(angle)
{
}

osg::Matrix ViewpointCycler::OrbitEffect ::update(float progress)
{
    double angle = m_angle * progress;

    osg::Vec3 diff = m_initial.getTrans() - m_center;

    auto trans = osg::Matrix::translate(diff);
    auto rot = osg::Matrix::rotate(angle, osg::Vec3(0, 0, 1));

    return (osg::Matrix::inverse(trans) * rot * trans) * m_initial;

    // osg::Vec3 pos = diff;
    // osg::Vec3 forward(0, 1, 0);
    // return m_initial * osg::Matrix::translate(forward * distance);
}

ViewpointCycler::ViewpointCycler()
{
    reset();
}

bool ViewpointCycler::advanceCurrentViewpoint()
{
    double deltaTime = cover->frameDuration();
    m_timeToNextViewpoint -= deltaTime;

    if (m_timeToNextViewpoint > 0)
        return false;

    int viewpointCount = getViewpoints().size();

    while (m_timeToNextViewpoint < 0 || m_currentViewpointIndex < 0)
    {
        m_timeToNextViewpoint += m_interval;
        m_currentViewpointIndex++;
        m_currentViewpointIndex %= viewpointCount;
    }

    return true;
}

void ViewpointCycler::advance(int count)
{
    m_currentViewpointIndex += count;
    m_currentViewpointIndex %= getViewpoints().size();
    m_timeToNextViewpoint = m_interval;
    applyChange();
}

void ViewpointCycler::reset()
{
    m_currentViewpointIndex = -1;
    m_timeToNextViewpoint = 0.0;
}

void ViewpointCycler::applyChange()
{
    auto viewpoint = getViewpoints()[m_currentViewpointIndex];
    viewpoint->activate(true);

    auto inv = cover->getObjectsXform()->getInverseMatrix();

    // TODO: configure effect per viewpoint
    switch (m_currentViewpointIndex % 2)
    {
    case 0:
        m_currentEffect = std::make_unique<DollyEffect>(inv, 20.0 * cover->getScale());
        break;
    case 1:
    default:
        m_currentEffect = std::make_unique<OrbitEffect>(inv, inv.getTrans() + osg::Vec3(0, -100 * cover->getScale(), -30), m_currentViewpointIndex % 4 < 2 ? 0.1 : -0.1);
        break;
    }
}

void ViewpointCycler::update()
{
    if (advanceCurrentViewpoint())
    {
        applyChange();
    }

    if (m_currentEffect)
    {
        double progress = std::clamp(1.0 - (m_timeToNextViewpoint / m_interval), 0.0, 1.0);
        auto inv = m_currentEffect->update(progress);
        cover->getObjectsXform()->setMatrix(osg::Matrix::inverse(inv));
    }
}
