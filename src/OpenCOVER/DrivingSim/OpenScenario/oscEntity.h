/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_ENTITY_H
#define OSC_ENTITY_H

#include "oscExport.h"
#include "oscNameRefId.h"
#include "oscObjectVariable.h"

#include "oscChoiceObject.h"
#include "oscChoiceController.h"
#include "oscObserverId.h"


namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscEntity: public oscNameRefId
{
public:
    oscEntity()
    {
        OSC_OBJECT_ADD_MEMBER(objectChoice, "oscChoiceObject");
        OSC_OBJECT_ADD_MEMBER(controllerChoice, "oscChoiceController");
        OSC_OBJECT_ADD_MEMBER(observer, "oscObserverId");
    };

    oscChoiceObjectMember objectChoice;
    oscChoiceControllerMember controllerChoice;
    oscObserverIdMember observer;
};

typedef oscObjectVariable<oscEntity *> oscEntityMember;

}

#endif //OSC_ENTITY_H
