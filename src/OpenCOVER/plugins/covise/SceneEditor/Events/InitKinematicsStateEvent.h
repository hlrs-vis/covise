/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef INIT_KINEMATICS_STATE_EVENT_H
#define INIT_KINEMATICS_STATE_EVENT_H

#include "Event.h"

#include <string>

class InitKinematicsStateEvent : public Event
{
public:
    InitKinematicsStateEvent();
    virtual ~InitKinematicsStateEvent();

    void setState(std::string state);
    std::string getState();

private:
    std::string _state;
};

#endif
