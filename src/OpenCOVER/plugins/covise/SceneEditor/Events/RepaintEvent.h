/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef REPAINT_EVENT_H
#define REPAINT_EVENT_H

#include "Event.h"

#include <iostream>

class SceneObject;

class RepaintEvent : public Event
{
public:
    RepaintEvent();
    virtual ~RepaintEvent();
};

#endif
