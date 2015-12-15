/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */
#ifndef OSC_PEDESTRIAN_CONTROLLER_H
#define OSC_PEDESTRIAN_CONTROLLER_H
#include <oscExport.h>
#include <oscObjectBase.h>
#include <oscObjectVariable.h>
#include <oscVariables.h>

namespace OpenScenario {


class OPENSCENARIOEXPORT motionType: public oscEnumType
{
public:
    static motionType *instance(); 
private:
    motionType();
    static motionType *inst;
};

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscPedestrianController: public oscObjectBase
{
public:

    enum motion
    {
        walk,
        jog,
        run,
        dead,
    };
    oscPedestrianController()
    {
       
		OSC_ADD_MEMBER(motion);
		motion.enumType = motionType::instance();
    };
	oscEnum motion;
};

typedef oscObjectVariable<oscPedestrianController *> oscPedestrianControllerMember;

}

#endif //OSC_PEDESTRIAN_CONTROLLER_H
