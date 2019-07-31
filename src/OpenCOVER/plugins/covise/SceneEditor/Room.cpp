/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "Room.h"
#include "Window.h"

#include "Events/TransformChangedEvent.h"
#include "Events/SetSizeEvent.h"

#include "SceneUtils.h"
#include "Settings.h"

#include <cover/coVRMSController.h>
#include <cover/VRSceneGraph.h>
#include <grmsg/coGRObjGeometryMsg.h>
#include <net/message.h>
#include <covise/covise_msg.h>

using namespace covise;
using namespace opencover;

// constructor and destructor
// ---------------------------------------------
Room::Room(std::string name, float width, float length, float height)
    : _length(length)
    , _width(width)
    , _height(height)
    , _name(name)
{
    _intersectionPlane = new coPlane(osg::Vec3(0.0, 0.0, 1.0), osg::Vec3(0.0, 0.0, 0.0));
    _type = SceneObjectTypes::ROOM;
}

Room::~Room()
{
    std::vector<Barrier *>::iterator it = _barriers.begin();
    while (it != _barriers.end())
    {
        delete (*it);
        _barriers.erase(it);
        it = _barriers.begin();
    }
    delete _intersectionPlane;
}

// ---------------------------------------------
EventErrors::Type Room::receiveEvent(Event *e)
{
    if (e->getType() == EventTypes::PRE_FRAME_EVENT)
    {
        for (std::vector<Barrier *>::iterator it = _barriers.begin(); it != _barriers.end(); it++)
            (*it)->preFrame();
    }
    else if (e->getType() == EventTypes::REPAINT_EVENT)
    {
        for (std::vector<Barrier *>::iterator it = _barriers.begin(); it != _barriers.end(); it++)
            (*it)->repaint();
    }
    else if (e->getType() == EventTypes::SET_SIZE_EVENT)
    {
        SetSizeEvent *sse = dynamic_cast<SetSizeEvent *>(e);
        setSize(sse->getWidth(), sse->getHeight(), sse->getLength());
    }
    else if (e->getType() == EventTypes::SETTINGS_CHANGED_EVENT)
    {
        std::vector<Barrier *>::iterator it = _barriers.begin();
        while (it != _barriers.end())
        {
            (*it)->setGridVisible(Settings::instance()->isGridVisible());
            ++it;
        }
    }

    return SceneObject::receiveEvent(e);
}

// setter and getter
// ---------------------------------------------
float Room::getLength()
{
    return _length;
}

float Room::getWidth()
{
    return _width;
}

float Room::getHeight()
{
    return _height;
}

void Room::setLength(float l)
{
    _length = l;
    // tell walls
    updateSize();
}

void Room::setWidth(float w)
{
    _width = w;
    // tell walls
    updateSize();
}

void Room::setHeight(float h)
{
    _height = h;
    // tell walls
    updateSize();
}

void Room::setSize(float w, float h, float l)
{
    _width = w;
    _height = h;
    _length = l;
    // tell walls
    updateSize();
}

float Room::getSize()
{
    float size = _width;
    if (_height > size)
        size = _height;
    if (_length > size)
        size = _length;
    return size;
}

void Room::addBarrier(Barrier *barrier)
{
    _barriers.push_back(barrier);
    barrier->setGridVisible(Settings::instance()->isGridVisible());
}

void Room::addWindow(Window *window)
{
    _windows.push_back(window);
}

void Room::removeWindow(Window *window)
{
    _windows.remove(window);
}

std::list<Window *> Room::getWindows()
{
    return _windows;
}

Floor *Room::getFloor()
{
    std::vector<Barrier *>::iterator it = _barriers.begin();
    while (it != _barriers.end())
    {
        if ((*it)->getAlignment() == Barrier::BOTTOM)
        {
            return (Floor *)(*it);
        }
        ++it;
    }
    return NULL;
}

Ceiling *Room::getCeiling()
{
    std::vector<Barrier *>::iterator it = _barriers.begin();
    while (it != _barriers.end())
    {
        if ((*it)->getAlignment() == Barrier::TOP)
        {
            return (Ceiling *)(*it);
        }
        ++it;
    }
    return NULL;
}

// note: for now we are assuming, the wall was positioned along it's normal
// 1: scale position to center so it's on the (infinite) wall
// 2: use wall if scaled position is within the walls width
Wall *Room::getClosestWall(osg::Vec3 position)
{
    Wall *backupWall = NULL;

    position -= getPosition();
    position[2] = 0.0f;

    std::vector<Barrier *>::iterator it = _barriers.begin();
    while (it != _barriers.end())
    {
        Wall *wall = dynamic_cast<Wall *>(*it);
        if (wall)
        {
            backupWall = wall;
            // get info
            osg::Vec3 w_norm = wall->getNormal();
            osg::Vec3 w_pos = wall->getPosition();
            w_pos[2] = 0.0f;
            // length of projection of position onto normal
            float proj_len = fabs(position * w_norm);
            // factor = ratio of wallOffset to length of p
            float f = 1.0f;
            if (proj_len != 0.0f)
            {
                f = w_pos.length() / proj_len;
            }
            // scale position
            osg::Vec3 scaledPosition = position * f;
            // check
            if ((w_pos - scaledPosition).length() <= wall->getWidth() / 2.0f)
            {
                return wall;
            }
        }
        ++it;
    }
    return backupWall;
}

// position room
// ---------------------------------------------
void Room::setPosition(osg::Vec3 pos)
{
    osg::MatrixTransform *transformNode = (osg::MatrixTransform *)getGeometryNode();
    osg::Matrix m = transformNode->getMatrix();
    m.makeTranslate(pos);
    transformNode->setMatrix(m);
}

osg::Vec3 Room::getPosition()
{
    osg::MatrixTransform *transformNode = (osg::MatrixTransform *)getGeometryNode();
    osg::Matrix m = transformNode->getMatrix();
    return m.getTrans();
}

void Room::updateSize()
{
    for (std::vector<Barrier *>::iterator it = _barriers.begin(); it != _barriers.end(); it++)
        (*it)->setSize(_width, _height, _length);
    TransformChangedEvent tce;
    tce.setSender(this);
    receiveEvent(&tce);
}

void Room::sendMessageGeometryToGUI()
{
    //std::cerr << "Room::sendMessageGeometryToGUI" << std::endl;
    // create Message
    if (coVRMSController::instance()->isMaster())
    {
        grmsg::coGRObjGeometryMsg geoMsg(getCoviseKey().c_str(), _width, _height, _length);
        Message grmsg{ COVISE_MESSAGE_UI, DataHandle{(char *)geoMsg.c_str(),strlen(geoMsg.c_str()) + 1 , false} };
        opencover::cover->sendVrbMessage(&grmsg);
    }
}

// ---------------------------------------------

//bool Room::isInside(osg::Vec3 pos, float tolerance)
//{
//   // NOTE: room rotation is ignored
//   osg::Vec3 roomPos = getPosition();
//   return ( (pos[0] >= roomPos[0] - _width/2.0f - tolerance) &&
//            (pos[0] <= roomPos[0] + _width/2.0f + tolerance) &&
//            (pos[1] >= roomPos[1] - _length/2.0f - tolerance) &&
//            (pos[1] <= roomPos[1] + _length/2.0f + tolerance) &&
//            (pos[2] >= roomPos[2] - tolerance) &&
//            (pos[2] <= roomPos[2] + _height + tolerance)    );
//}

//osg::Vec3 Room::getIntersectionWorld()
//{
//   for (std::vector<Barrier*>::iterator it=_barriers.begin(); it!=_barriers.end(); it++)
//   {
//      osg::Vec3 n = (*it)->getNormal();
//      n = opencover::cover->getXformMat().getRotate() * n;
//
//      osg::Vec3 p = (*it)->getPosition();
//      p *= opencover::VRSceneGraph::instance()->scaleFactor();
//      p = opencover::cover->getXformMat().preMult(p);
//
//      // check if we are looking at the front side of the plane
//      if (n[1] < 0.0f)
//      {
//         _intersectionPlane->update(n*-1.0f, p);
//         osg::Vec3 planeIntersectionPoint;
//         if (SceneUtils::getPlaneIntersection(_intersectionPlane, opencover::cover->getPointerMat(), planeIntersectionPoint))
//         {
//            // remove cover world matrix transform effects
//            planeIntersectionPoint = osg::Matrixd::inverse(opencover::cover->getXformMat()).preMult(planeIntersectionPoint);
//            planeIntersectionPoint /= opencover::VRSceneGraph::instance()->scaleFactor();
//            if (isInside(planeIntersectionPoint, 1.0f))
//            {
//               return planeIntersectionPoint;
//            }
//         }
//
//      }
//   }
//
//   return osg::Vec3(0.0f, 0.0f, 0.0f);
//}
