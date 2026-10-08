/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */
#ifndef VIEWPOINT_CYCLER_H
#define VIEWPOINT_CYCLER_H

#include <osg/MatrixTransform>
#include <osg/Matrix>
#include <cover/coVRPluginSupport.h>

#include <cover/coVRPlugin.h>
#include <OpenVRUI/osg/mathUtils.h>

#include <cover/ui/Owner.h>

using namespace opencover;

class ViewpointCycler
{
public:
    class Effect
    {
    public:
        Effect(osg::Matrix initial);
        virtual osg::Matrix update(float progress);

    protected:
        osg::Matrix m_initial;
    };

    class DollyEffect : public Effect
    {
    public:
        DollyEffect(osg::Matrix initial, double distance);
        virtual osg::Matrix update(float progress) override;

    protected:
        double m_distance;
    };

    class OrbitEffect : public Effect
    {
    public:
        OrbitEffect(osg::Matrix initial, osg::Vec3 center, double angle = M_PI / 2);
        virtual osg::Matrix update(float progress) override;

    protected:
        osg::Vec3 m_center;
        double m_angle;
    };

    ViewpointCycler();

    void update();
    void reset();

    void advance(int count);

private:
    void applyChange();
    bool advanceCurrentViewpoint();

    double m_interval = 5.0;

    // Initialized by reset()
    int m_currentViewpointIndex;
    double m_timeToNextViewpoint;

    std::unique_ptr<Effect> m_currentEffect;
};

#endif
