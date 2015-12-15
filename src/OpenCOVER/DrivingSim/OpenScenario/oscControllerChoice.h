/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */
#ifndef OSC_CONTROLLER_CHOICE_H
#define OSC_CONTROLLER_CHOICE_H
#include <oscExport.h>
#include <oscObjectBase.h>
#include <oscObjectVariable.h>
#include <oscVariables.h>
#include <oscDriverRef.h>
#include <oscPedestrianController.h>

namespace OpenScenario {


/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscControllerChoice: public oscObjectBase
{
public:
    oscControllerChoice()
    {
        OSC_OBJECT_ADD_MEMBER(OSCDriverReference,"oscDriverRef");
		OSC_OBJECT_ADD_MEMBER(PedestrianController, "oscPedestrianController");
    };
	oscDriverRefMember OSCDriverReference;
    oscPedestrianControllerMember PedestrianController;
};

typedef oscObjectVariable<oscControllerChoice *> oscControllerChoiceMember;

}

#endif //OSC_CONTROLLER_CHOICE_H
