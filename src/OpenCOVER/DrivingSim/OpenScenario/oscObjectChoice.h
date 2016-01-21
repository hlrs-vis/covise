/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_OBJECT_CHOICE_H
#define OSC_OBJECT_CHOICE_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"

#include "oscVehicleRef.h"
#include "oscPedestrianRef.h"
#include "oscMiscObjectRef.h"


namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscObjectChoice: public oscObjectBase
{
public:
    oscObjectChoice()
    {
        OSC_OBJECT_ADD_MEMBER(OSCVehicle, "oscVehicleRef");
        OSC_OBJECT_ADD_MEMBER(OSCPedestrian, "oscPedestrianRef");
        OSC_OBJECT_ADD_MEMBER(OSCMiscObject, "oscMiscObjectRef");
    };

    oscVehicleRefMember OSCVehicle;
    oscPedestrianRefMember OSCPedestrian;
    oscMiscObjectRefMember OSCMiscObject;
};

typedef oscObjectVariable<oscObjectChoice *> oscObjectChoiceMember;

}

#endif //OSC_OBJECT_CHOICE_H
