/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */
#ifndef OSC_VEHICLE_REF_H
#define OSC_VEHICLE_REF_H
#include <oscExport.h>
#include <oscObjectBase.h>
#include <oscObjectVariable.h>
#include <oscNameId.h>

namespace OpenScenario {


/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscVehicleRef: public oscObjectBase
{
public:
    oscVehicleRef()
    {
        OSC_OBJECT_ADD_MEMBER(name,"oscNameId");
    };
    oscNameIdMember name;
};

typedef oscObjectVariable<oscVehicleRef *> oscVehicleRefMember;

}

#endif //OSC_VEHICLE_REF_H
