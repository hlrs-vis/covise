/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef ROOM_H
#define ROOM_H

#include "SceneObject.h"
#include "Wall.h"
#include "Floor.h"
#include "Ceiling.h"

#include <osg/MatrixTransform>

#include <PluginUtil/coPlane.h>

class Window;

class Room : public SceneObject
{
public:
    Room(std::string name = "", float width = 3000.0, float length = 4000.0, float height = 2500.0);
    virtual ~Room();

    virtual EventErrors::Type receiveEvent(Event *e);

    // setter and getter
    float getLength();
    float getWidth();
    float getHeight();
    void setLength(float l);
    void setWidth(float w);
    void setHeight(float h);
    void setSize(float w, float h, float l);
    float getSize();

    void addBarrier(Barrier *);
    void addWindow(Window *);
    void removeWindow(Window *);
    std::list<Window *> getWindows();

    Floor *getFloor();
    Ceiling *getCeiling();
    Wall *getClosestWall(osg::Vec3 pos);

    //position room
    void setPosition(osg::Vec3 pos);
    osg::Vec3 getPosition();

    // toggleInteracotrs
    void toggleInteractors();
    void sendMessageGeometryToGUI();

private:
    float _length;
    float _width;
    float _height;
    std::string _name;

    // barriers of the room
    std::vector<Barrier *> _barriers;
    // list of windows
    std::list<Window *> _windows;

    void updateSize();

    opencover::coPlane *_intersectionPlane;
};

#endif
