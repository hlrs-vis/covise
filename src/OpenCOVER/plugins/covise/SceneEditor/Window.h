/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef WINDOW_H
#define WINDOW_H

#include <osg/ShapeDrawable>

#include "SceneObject.h"

class Room;

class Window : public SceneObject
{
public:
    Window();
    virtual ~Window();

    EventErrors::Type receiveEvent(Event *e);

    void setGeometry(float w, float h);
    float getWidth();
    float getHeight();

    osg::BoundingBox getRotatedBBox();

private:
    osg::ref_ptr<osg::Box> _frameBox[4];
    osg::ref_ptr<osg::ShapeDrawable> _frameDrawable[4];
    osg::ref_ptr<osg::Geode> _frameGeode;
    osg::ref_ptr<osg::Box> _surfaceBox;
    osg::ref_ptr<osg::ShapeDrawable> _surfaceDrawable;
    osg::ref_ptr<osg::Geode> _surfaceGeode;

    float _width, _height;

    void _updateGeometry();
    void _sendRepaintEvent(Room *room = NULL);
};

#endif
