/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "Window.h"
#include "Events/RepaintEvent.h"
#include "Events/MountEvent.h"
#include "Events/UnmountEvent.h"
#include "Events/SetSizeEvent.h"
#include "Events/TransformChangedEvent.h"
#include "Room.h"

#include <cover/coVRPluginSupport.h>

#include <osg/Material>

#define BORDER_WIDTH 20.0f

Window::Window()
{
    _name = "";
    _type = SceneObjectTypes::WINDOW;
    _frameGeode = NULL;

    _width = 1.0f;
    _height = 1.0f;
}

Window::~Window()
{
}

void Window::setGeometry(float w, float h)
{
    _width = w;
    _height = h;
    _updateGeometry();

    TransformChangedEvent tce;
    tce.setSender(this);
    receiveEvent(&tce);
}

float Window::getWidth()
{
    return _width;
}

float Window::getHeight()
{
    return _height;
}

EventErrors::Type Window::receiveEvent(Event *e)
{
    if (e->getType() == EventTypes::SET_SIZE_EVENT)
    {
        SetSizeEvent *sse = dynamic_cast<SetSizeEvent *>(e);
        setGeometry(sse->getWidth(), sse->getHeight());
    }
    else if (e->getType() == EventTypes::MOUNT_EVENT)
    {
        // add Window to Room
        MountEvent *me = dynamic_cast<MountEvent *>(e);
        Room *room = dynamic_cast<Room *>(me->getMaster());
        if (room)
        {
            room->addWindow(this);
            _sendRepaintEvent();
        }
    }
    else if (e->getType() == EventTypes::UNMOUNT_EVENT)
    {
        // remove Window from room and tell room to repaint
        UnmountEvent *ue = dynamic_cast<UnmountEvent *>(e);
        Room *room = dynamic_cast<Room *>(ue->getMaster());
        if (room)
        {
            room->removeWindow(this);
            _sendRepaintEvent(room);
        }
    }
    else if (e->getType() == EventTypes::TRANSFORM_CHANGED_EVENT)
    {
        _sendRepaintEvent();
    }

    return SceneObject::receiveEvent(e);
}

osg::BoundingBox Window::getRotatedBBox()
{
    osg::Vec3 extent = osg::Vec3(_width / 2.0f, 0.0f, _height / 2.0f);
    extent = getRotate() * extent;
    osg::BoundingBox bbox(-fabs(extent[0]), -fabs(extent[1]), -fabs(extent[2]), fabs(extent[0]), fabs(extent[1]), fabs(extent[2]));
    return bbox;
}

void Window::_updateGeometry()
{
    // FRAME
    if (!_frameGeode)
    {
        osg::Group *parent = dynamic_cast<osg::Group *>(getGeometryNode());
        _frameGeode = new osg::Geode();
        parent->addChild(_frameGeode.get());
        for (int i = 0; i < 4; ++i)
        {
            _frameDrawable[i] = new osg::ShapeDrawable();
            _frameGeode->addDrawable(_frameDrawable[i].get());
            _frameBox[i] = new osg::Box();
            _frameDrawable[i]->setShape(_frameBox[i].get());
        }
        // material
        osg::StateSet *sset = _frameGeode->getOrCreateStateSet();
        osg::Material *material = new osg::Material();
        sset->setAttributeAndModes(material, osg::StateAttribute::ON);
    }
    // bottom
    _frameBox[0]->setHalfLengths(osg::Vec3(_width / 2.0f + 2.0f * BORDER_WIDTH, BORDER_WIDTH, BORDER_WIDTH));
    _frameBox[0]->setCenter(osg::Vec3(0.0f, 0.0f, -_height / 2.0f - BORDER_WIDTH));
    // top
    _frameBox[1]->setHalfLengths(osg::Vec3(_width / 2.0f + 2.0f * BORDER_WIDTH, BORDER_WIDTH, BORDER_WIDTH));
    _frameBox[1]->setCenter(osg::Vec3(0.0f, 0.0f, _height / 2.0f + BORDER_WIDTH));
    // left
    _frameBox[2]->setHalfLengths(osg::Vec3(BORDER_WIDTH, BORDER_WIDTH, _height / 2.0f));
    _frameBox[2]->setCenter(osg::Vec3(-_width / 2.0f - BORDER_WIDTH, 0.0f, 0.0f));
    // right
    _frameBox[3]->setHalfLengths(osg::Vec3(BORDER_WIDTH, BORDER_WIDTH, _height / 2.0f));
    _frameBox[3]->setCenter(osg::Vec3(_width / 2.0f + BORDER_WIDTH, 0.0f, 0.0f));
    for (int i = 0; i < 4; ++i)
    {
        _frameDrawable[i]->dirtyDisplayList();
        _frameDrawable[i]->dirtyBound();
    }

    // SURFACE
    if (!_surfaceGeode)
    {
        osg::Group *parent = dynamic_cast<osg::Group *>(getGeometryNode());
        _surfaceGeode = new osg::Geode();
        parent->addChild(_surfaceGeode.get());
        _surfaceDrawable = new osg::ShapeDrawable();
        _surfaceGeode->addDrawable(_surfaceDrawable.get());
        _surfaceBox = new osg::Box();
        _surfaceDrawable->setShape(_surfaceBox.get());
        // material
        osg::StateSet *sset = _surfaceGeode->getOrCreateStateSet();
        sset->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);
        sset->setMode(GL_BLEND, osg::StateAttribute::ON);
        osg::Material *material = new osg::Material();
        material->setTransparency(osg::Material::FRONT_AND_BACK, 1.0f);
        sset->setAttributeAndModes(material, osg::StateAttribute::ON);
    }
    _surfaceBox->setHalfLengths(osg::Vec3(_width / 2.0f, 1.0f, _height / 2.0f));
    _surfaceDrawable->dirtyDisplayList();
    _surfaceDrawable->dirtyBound();
}

void Window::_sendRepaintEvent(Room *room)
{
    if (!room)
    {
        room = dynamic_cast<Room *>(getParent());
    }
    if (room)
    {
        RepaintEvent re;
        re.setSender(this);
        room->receiveEvent(&re);
    }
}
