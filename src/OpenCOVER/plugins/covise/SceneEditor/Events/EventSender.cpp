/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "EventSender.h"

#include "../Behaviors/Behavior.h"

EventSender::EventSender()
{
    _so = NULL;
    _be = NULL;
}

EventSender::EventSender(SceneObject *so)
{
    _so = so;
    _be = NULL;
}

EventSender::EventSender(Behavior *be)
{
    if (be)
    {
        _so = be->getSceneObject();
    }
    _be = be;
}

EventSender::~EventSender()
{
}

bool EventSender::isNull()
{
    return (_so == NULL);
}

bool EventSender::isSceneObject()
{
    return (_so && !_be);
}

bool EventSender::isBehavior()
{
    return (_be != NULL);
}

SceneObject *EventSender::getSceneObject()
{
    return _so;
}

Behavior *EventSender::getBehavior()
{
    return _be;
}
