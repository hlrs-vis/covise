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

class OPENSCENARIOEXPORT partnerTObjTypeType: public oscEnumType
{
public:
    static partnerTObjTypeType *instance();
private:
    partnerTObjTypeType();
    static partnerTObjTypeType *inst;
};

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscPartnerType: public oscObjectBase
{
public:
    oscPartnerType()
    {
        OSC_ADD_MEMBER(objectType);

        objectType.enumType = partnerTObjTypeType::instance();
    };

    oscEnum objectType;

    enum partnerTObjType
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
