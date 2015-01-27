/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef EVENT_SENDER_H
#define EVENT_SENDER_H

// forward declaration
class SceneObject;
class Behavior;

class EventSender
{
public:
    EventSender();
    EventSender(SceneObject *so);
    EventSender(Behavior *be);

    virtual ~EventSender();

    bool isNull();
    bool isSceneObject();
    bool isBehavior();

    SceneObject *getSceneObject(); // can also be used if sender is a behavior
    Behavior *getBehavior();

protected:
    SceneObject *_so;
    Behavior *_be;
};

#endif
