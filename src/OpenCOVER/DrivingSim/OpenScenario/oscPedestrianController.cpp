/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#include <oscPedestrianController.h>


using namespace OpenScenario;


motionType::motionType()
{
    addEnum("walk", oscPedestrianController::walk);
    addEnum("jog", oscPedestrianController::jog);
    addEnum("run", oscPedestrianController::run);
    addEnum("dead", oscPedestrianController::dead);
}

motionType *motionType::instance()
{
    if(inst == NULL)
    {
        inst = new motionType();
    }
    return inst;
}

motionType *motionType::inst = NULL;
