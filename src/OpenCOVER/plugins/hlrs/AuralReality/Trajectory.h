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
#include <grpc/grpc.h>
#include <grpcpp/channel.h>

#include "CustomTransformInteractor.h"
#include "Selection.h"

class Trajectory;
class TrajectoryPoint : public Selectable
{
    friend class Trajectory;

public:
    TrajectoryPoint(Trajectory *trajectory_);
    ~TrajectoryPoint();
    void preFrame();

    void setTransforms(const osg::Matrix &anchor_, const osg::Vec3 &controlPointIn_, const osg::Vec3 &controlPointOut_);

protected:
    virtual void updateSelection() override;

private:
    Trajectory *trajectory;

    osg::ref_ptr<osg::Group> groupNode;
    osg::ref_ptr<osg::MatrixTransform> anchorNode;
    osg::ref_ptr<osg::MatrixTransform> controlPointInNode;
    osg::ref_ptr<osg::MatrixTransform> controlPointOutNode;

    CustomTransformInteractor anchorInteractor;
    CustomTransformInteractor controlPointInInteractor;
    CustomTransformInteractor controlPointOutInteractor;

    osg::Matrix anchor;
    osg::Vec3 controlPointIn;
    osg::Vec3 controlPointOut;

    SelectableSensor *sensor;
};

class Trajectory : public Selectable
{

public:
    Trajectory();
    ~Trajectory();
    void preFrame()
    {
        for (auto &p : points)
        {
            p->preFrame();
        }
    }

    virtual void updateSelection()
    {
        for (auto &p : points)
        {
            if (isSelected())
                p->select();
            else
                p->deselect();
        }
    }

    void pointChanged();
    void rebuildGeometry();

    std::vector<std::shared_ptr<TrajectoryPoint>> points;
    osg::ref_ptr<osg::Geode> linesGeode;
    osg::ref_ptr<osg::Geometry> linesGeometry;
    osg::ref_ptr<osg::Geometry> bezierGeometry;
};

#endif
