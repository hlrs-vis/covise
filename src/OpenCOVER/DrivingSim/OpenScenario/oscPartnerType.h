/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_PARTNER_TYPE_H
#define OSC_PARTNER_TYPE_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"

#include "oscVariables.h"


namespace OpenScenario {

class OPENSCENARIOEXPORT objectTypePartnerType: public oscEnumType
{
public:
    static objectTypePartnerType *instance();
private:
    objectTypePartnerType();
    static objectTypePartnerType *inst;
};

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscPartnerType: public oscObjectBase
{
public:
    oscPartnerType()
    {
        OSC_ADD_MEMBER(objectType);

        objectType.enumType = objectTypePartnerType::instance();
    };

    oscEnum objectType;

    enum objectTypePartner
    {
        vehicle,
        pedestrian,
        trafficSign,
        infrastructure,
    };
};

typedef oscObjectVariable<oscPartnerType *> oscPartnerTypeMember;

}

#endif //OSC_PARTNER_TYPE_H
