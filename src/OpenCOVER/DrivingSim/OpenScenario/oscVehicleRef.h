/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_VEHICLE_REF_H
#define OSC_VEHICLE_REF_H

#include <oscExport.h>
#include <oscNameRefId.h>
#include <oscObjectVariable.h>


namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscVehicleRef: public oscNameRefId
{
public:
    oscVehicleRef()
    {

    };
};

typedef oscObjectVariable<oscVehicleRef *> oscVehicleRefMember;

}

#endif //OSC_VEHICLE_REF_H
