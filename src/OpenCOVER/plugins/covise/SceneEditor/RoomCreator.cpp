/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "RoomCreator.h"

#include <QDir>
#include <iostream>
#include <osgDB/ReadFile>
#include <cover/coVRPluginSupport.h>
#include <cover/VRSceneGraph.h>

#include "SceneEditor.h"
#include "SceneObjectManager.h"
#include "Ceiling.h"
#include "Floor.h"
#include "Wall.h"
using namespace covise;
using namespace opencover;

RoomCreator::RoomCreator()
{
}

RoomCreator::~RoomCreator()
{
}

SceneObject *RoomCreator::createFromXML(QDomElement *root)
{
    Room *room = new Room();
    if (!buildFromXML(room, root))
    {
        delete room;
        return NULL;
    }
    return room;
}

bool RoomCreator::buildFromXML(SceneObject *so, QDomElement *root)
{
    // create geometry
    if (!buildGeometryFromXML((Room *)so, root))
    {
        return false;
    }
    return SceneObjectCreator::buildFromXML(so, root);
}

bool RoomCreator::buildGeometryFromXML(Room *so, QDomElement *root)
{
    QDomElement geoElem = root->firstChildElement("geometry");
    if (!geoElem.isNull())
    {
        QDomElement widthEle = geoElem.firstChildElement("width");
        QDomElement heightEle = geoElem.firstChildElement("height");
        QDomElement lengthEle = geoElem.firstChildElement("length");
        float width = so->getWidth();
        float height = so->getHeight();
        float length = so->getLength();
        if (!widthEle.isNull())
            width = widthEle.attribute("value").toFloat();
        if (!heightEle.isNull())
            height = heightEle.attribute("value").toFloat();
        if (!lengthEle.isNull())
            length = lengthEle.attribute("value").toFloat();

        // for positioning room
        so->setGeometryNode(new osg::MatrixTransform());

        // create walls
        so->addBarrier(new Ceiling(so, "Ceiling", width, length));
        so->addBarrier(new Floor(so, "Floor", width, length));
        so->addBarrier(new Wall(so, "WallFront", width, height, Wall::FRONT));
        so->addBarrier(new Wall(so, "WallLeft", length, height, Wall::LEFT));
        so->addBarrier(new Wall(so, "WallBack", width, height, Wall::BACK));
        so->addBarrier(new Wall(so, "WallRight", length, height, Wall::RIGHT));

        so->setSize(width, height, length);
        //so->toggleInteractors();

        // if it's not the first room, transform it
        std::vector<SceneObject *> rooms = SceneObjectManager::instance()->getSceneObjectsOfType(SceneObjectTypes::ROOM);
        float xPos = 0;
        for (std::vector<SceneObject *>::iterator it = rooms.begin(); it != rooms.end(); it++)
        {
            Room *r = dynamic_cast<Room *>(*it);
            xPos = xPos + r->getWidth() + 300;
        }
        so->setPosition(osg::Vec3(xPos, 0, 0));

        // finally: switch to top view and make "View All"
        opencover::cover->sendMessage(SceneEditor::plugin, "ViewPoint", 0, 17, "loadViewpoint top");
        opencover::VRSceneGraph::instance()->viewAll();
    }

    return true;
}
