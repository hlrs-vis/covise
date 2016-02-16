/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_CHOICE_CONTROLLER_H
#define OSC_CHOICE_CONTROLLER_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"

#include "oscNameRefId.h"
#include "oscPedestrianController.h"


namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscChoiceController: public oscObjectBase
{
public:
    oscChoiceController()
    {
        OSC_OBJECT_ADD_MEMBER(driverReference, "oscNameRefId");
        OSC_OBJECT_ADD_MEMBER(pedestrianController, "oscPedestrianController");
    };

    oscNameRefIdMember driverReference;
    oscPedestrianControllerMember pedestrianController;
};

typedef oscObjectVariable<oscChoiceController *> oscChoiceControllerMember;

}

#endif //OSC_CHOICE_CONTROLLER_H
