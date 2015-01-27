/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "Behavior.h"

#include <iostream>

#include "../SceneObject.h"

Behavior::Behavior()
{
    _type = BehaviorTypes::NONE;
    _isEnabled = true;
}

Behavior::~Behavior()
{
}

int Behavior::attach(SceneObject *so)
{
    if (so == NULL)
    {
        std::cerr << "Trying to attach Behavior to NULL SceneObject!" << std::endl;

        return -1;
    }

    _sceneObject = so;

    return 1;
}

int Behavior::detach()
{
    _sceneObject = NULL;

    return 1;
}

EventErrors::Type Behavior::receiveEvent(Event *e)
{
    (void)e;

    return EventErrors::UNHANDLED;
}

bool Behavior::buildFromXML(QDomElement *behaviorElement)
{
    (void)behaviorElement;

    return true;
}

BehaviorTypes::Type Behavior::getType()
{
    return _type;
}

SceneObject *Behavior::getSceneObject()
{
    return _sceneObject;
}

void Behavior::setEnabled(bool b)
{
    _isEnabled = b;
}

bool Behavior::isEnabled()
{
    return _isEnabled;
}
