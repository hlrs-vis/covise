/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCEXTENT_H
#define OSCEXTENT_H

#include "../oscExport.h"
#include "../oscObjectBase.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT Enum_domain_time_distanceType : public oscEnumType
{
public:
static Enum_domain_time_distanceType *instance();
    private:
		Enum_domain_time_distanceType();
	    static Enum_domain_time_distanceType *inst; 
};
class OPENSCENARIOEXPORT oscExtent : public oscObjectBase
{
public:
oscExtent()
{
        OSC_ADD_MEMBER(value);
        OSC_ADD_MEMBER(domain);
        domain.enumType = Enum_domain_time_distanceType::instance();
    };
    oscDouble value;
    oscEnum domain;

    enum Enum_domain_time_distance
    {
time,
distance,

    };

};

typedef oscObjectVariable<oscExtent *> oscExtentMember;
typedef oscObjectVariableArray<oscExtent *> oscExtentArrayMember;


}

#endif //OSCEXTENT_H
