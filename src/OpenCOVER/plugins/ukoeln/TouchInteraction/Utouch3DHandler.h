/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// Utouch3DHandler.h

#pragma once

#include "TouchHandler.h"

#include <osg/Vec2>
#include <osg/Vec3>
#include <osg/Vec4>
#include <osg/Matrix>

#include <map>

//--------------------------------------------------------------------------------------------------
//
//--------------------------------------------------------------------------------------------------
class Utouch3DHandler : public TouchHandler
{
public:
    Utouch3DHandler();

    virtual ~Utouch3DHandler();

    virtual void init();

    virtual void finish();

    virtual void onTouchPressed(Touches const &touches, Touch const &reason);

    virtual void onTouchesMoved(Touches const &touches);

    virtual void onTouchReleased(Touches const &touches, Touch const &reason);

    virtual void onUpdate();

private:
    // Sets the down coordinates of all touch points to the current coordinates,
    // updates the rootTransform and updates the camera frustum
    void updateInternalState();

    void handleOneTouchPoint();
    void handleTwoTouchPoints();
    void handleThreeTouchPoints();

private:
    struct Cursor
    {
        // Touch point id
        int id;
        // Touch point position when pressed
        osg::Vec2 down;
        // Current touch point position
        osg::Vec2 curr;

        Cursor()
            : id(-1)
            , down(0, 0)
            , curr(0, 0)
        {
        }

        Cursor(int id, float x, float y)
            : id(id)
            , down(x, y)
            , curr(x, y)
        {
        }
    };

    typedef std::map<int /*touch point id*/, Cursor> CursorMap;

    CursorMap cursors;

    struct Frustum
    {
        double left;
        double right;
        double top;
        double bottom;
        double znear;
        double zfar;
    };

    Frustum frustum;

    osg::Matrix rootTransform;
};
