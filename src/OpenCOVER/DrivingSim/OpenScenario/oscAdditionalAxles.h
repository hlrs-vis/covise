/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_ADDITIONAL_AXLES_H
#define OSC_ADDITIONAL_AXLES_H

#include <oscExport.h>
#include <oscObjectBase.h>
#include <oscObjectArrayVariable.h>

#include <oscVehicleAxle.h>


namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscAdditionalAxles: public oscObjectBase
{
public:
    oscAdditionalAxles()
    {
        OSC_OBJECT_ADD_MEMBER(additional, "oscVehicleAxle");
    };

    oscVehicleAxleMember additional;
};

typedef oscObjectArrayVariable<oscAdditionalAxles *> oscAdditionalAxlesArrayMember;

}

#endif /* OSC_ADDITIONAL_AXLES_H */
