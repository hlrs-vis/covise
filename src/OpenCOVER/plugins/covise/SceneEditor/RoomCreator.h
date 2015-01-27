/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef ROOM_CREATOR
#define ROOM_CREATOR

#include "SceneObject.h"
#include "SceneObjectCreator.h"
#include "Room.h"

#include <QDomElement>

class RoomCreator : public SceneObjectCreator
{
public:
    RoomCreator();
    virtual ~RoomCreator();

    virtual SceneObject *createFromXML(QDomElement *root);

protected:
    virtual bool buildFromXML(SceneObject *so, QDomElement *root);

private:
    bool buildGeometryFromXML(Room *so, QDomElement *root);
};

#endif
