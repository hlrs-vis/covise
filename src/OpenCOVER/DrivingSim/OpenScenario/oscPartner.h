/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */
#ifndef OSC_PARTNER_H
#define OSC_PARTNER_H
#include <oscExport.h>
#include <oscObjectBase.h>
#include <oscObjectVariable.h>
#include <oscVariables.h>

namespace OpenScenario {


class OPENSCENARIOEXPORT objectTypeType: public oscEnumType
{
public:
    static objectTypeType *instance(); 
private:
    objectTypeType();
    static objectTypeType *inst;
};

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscPartner: public oscObjectBase
{
public:
	oscString object;
    enum objectType
    {
        vehicle,
        pedestrian,
        trafficSign,
        infrastructure,
    };
    oscPartner()
    {
		OSC_ADD_MEMBER(object);
		OSC_ADD_MEMBER(objectType);
		objectType.enumType = objectTypeType::instance();
    };
	oscEnum objectType;
};

typedef oscObjectVariable<oscPartner *> oscPartnerMember;

}

#endif //OSC_PARTNER_H
