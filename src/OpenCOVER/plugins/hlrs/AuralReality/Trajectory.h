/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _AURAL_REALITY_TRAJECTORY_H
#define _AURAL_REALITY_TRAJECTORY_H

#include <PluginUtil/coSensor.h>
#include <cover/coInteractor.h>
#include <cover/coVRPlugin.h>
#include <cover/ui/Owner.h>
#include <osg/MatrixTransform>
#include <osg/ShapeDrawable>
#include <osg/Geode>
#include <map>
#include <memory>
#include <string>

#include "CustomTransformInteractor.h"
#include "Selection.h"
#include "TmtEntity.h"

class Trajectory;
class TrajectoryPoint : public Selectable
{
    friend class Trajectory;

public:
    TrajectoryPoint(Trajectory *trajectory_);
    ~TrajectoryPoint();
    bool checkForChanges();

    void setTransforms(const osg::Vec3 &anchor_, const osg::Vec3 &controlPointIn_, const osg::Vec3 &controlPointOut_, const osg::Quat &rotation_);
    void setTime(double time);

    const osg::Vec3 getAnchor() const { return anchor; }
    const osg::Quat getRotation() const { return rotation; }
    const osg::Vec3 getControlPointIn() const { return controlPointIn; }
    const osg::Vec3 getControlPointOut() const { return controlPointOut; }

protected:
    virtual void updateSelection() override;

    void updateNodeTransforms(bool includeScaleInteractor = true);

private:
    Trajectory *trajectory;

    osg::ref_ptr<osg::Group> groupNode;
    osg::ref_ptr<osg::MatrixTransform> anchorNode;
    osg::ref_ptr<osg::MatrixTransform> controlPointInNode;
    osg::ref_ptr<osg::MatrixTransform> controlPointOutNode;

    CustomTransformInteractor anchorInteractor;
    CustomTransformInteractor controlPointInInteractor;
    CustomTransformInteractor controlPointOutInteractor;
    CustomTransformInteractor scaleInteractor;

    double m_time;

    // The control points are locations relative to the anchor location in 3D
    // space. The rotation affects neither of the points and only gives
    // directionality of the moving entity (usually a sound).
    osg::Vec3 anchor;
    osg::Vec3 controlPointIn;
    osg::Vec3 controlPointOut;
    osg::Quat rotation;

    float scaleInteractorDistance;

    SelectableSensor *sensor;
};

class Trajectory : public Selectable, public TmtEntity, public TmtEntityTransformMixin
{

public:
    Trajectory(const std::string &id);
    ~Trajectory();
    void preFrame();

    virtual void updateSelection() override
    {
        // for (auto &p : points)
        // {
        //     if (isSelected())
        //         p->select();
        //     else
        //         p->deselect();
        // }
    }

    void rebuildGeometry();

    std::vector<std::shared_ptr<TrajectoryPoint>> points;
    osg::ref_ptr<osg::Geode> linesGeode;
    osg::ref_ptr<osg::Geometry> linesGeometry;
    osg::ref_ptr<osg::Geometry> bezierGeometry;
    bool closed = false;
};

#endif
