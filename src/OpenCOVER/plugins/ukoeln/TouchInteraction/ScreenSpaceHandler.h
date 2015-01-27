/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// ScreenSpaceHandler.h
//
// See:
// Jason L. Reisman, Philip L. Davidson, and Jefferson Y. Han. 2009.
// A screen-space formulation for 2D and 3D direct manipulation.
// In Proceedings of the 22nd annual ACM symposium on User interface software
// and technology (UIST '09). ACM, New York, NY, USA, 69-78.
//

#pragma once

#include "TouchHandler.h"

#include <osg/Vec2>
#include <osg/Vec3>
#include <osg/Vec4>
#include <osg/Matrix>
#include <osg/Quat>

#include <osg/Camera>

#include <map>

class ScreenSpaceHandler
    : public TouchHandler
{
public:
    ScreenSpaceHandler()
    {
    }

    virtual ~ScreenSpaceHandler()
    {
    }

    virtual void init();

    virtual void finish();

    virtual void onTouchPressed(Touches const &touches, Touch const &reason);

    virtual void onTouchesMoved(Touches const &touches);

    virtual void onTouchReleased(Touches const &touches, Touch const &reason);

    virtual void onUpdate();

private:
    struct Cursor
    {
        // Touch point id
        int id;
        // Object-space position of touch point
        osg::Vec3d objectPos;
        // Screen-space target position of touch point
        osg::Vec2d targetPos;

        Cursor()
            : id(-1)
            , objectPos(0, 0, 0)
            , targetPos(0.5, 0.5)
        {
        }
    };

    typedef std::map<int /*touch point id*/, Cursor> CursorMap;

private:
    static bool intersect(osg::Vec3d &worldIntersectionPoint, osg::Vec2d const &screenPos);

    static void getTransformsFromState(double const *q, osg::Vec3 &T, osg::Quat &R);

    static void errorFunction(double *q, double *x, int m, int n, void *context);

    void computeTransform(osg::Vec3 &T, osg::Quat &R);

private:
    CursorMap cursors;

    osg::ref_ptr<osg::Camera> camera;

    osg::Matrix currentTransformMatrix;
};
