/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef BEHAVIOR_H
#define BEHAVIOR_H

#include "BehaviorTypes.h"
#include "../Events/Event.h"
#include "../ErrorCodes.h"

#include <QDomElement>

// forward declaration
class SceneObject;

class Behavior
{
public:
    Behavior();
    virtual ~Behavior();

    // must be called from SceneObject, when Behavior is attached/detached to it
    virtual int attach(SceneObject *);
    virtual int detach();

    virtual EventErrors::Type receiveEvent(Event *e);

    virtual BehaviorTypes::Type getType();

    virtual bool buildFromXML(QDomElement *behaviorElement);

    virtual void setEnabled(bool b);
    virtual bool isEnabled();

    SceneObject *getSceneObject();

protected:
    // the scene object this behavior belongs to
    SceneObject *_sceneObject;

    BehaviorTypes::Type _type;

    bool _isEnabled;
};

#endif
