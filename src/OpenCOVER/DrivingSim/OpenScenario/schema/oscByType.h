/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCBYTYPE_H
#define OSCBYTYPE_H

#include "../oscExport.h"
#include "../oscObjectBase.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscObjectTypeType : public oscEnumType
{
public:
static oscObjectTypeType *instance();
    private:
		oscObjectTypeType();
	    static oscObjectTypeType *inst; 
};
class OPENSCENARIOEXPORT oscByType : public oscObjectBase
{
public:
oscByType()
{
        OSC_ADD_MEMBER(type, 0);
    };
        const char *getScope(){return "/OSCCondition/ByEntity/EntityCondition/Collision";};
    oscEnum type;

    enum oscObjectType
    {
pedestrian,
vehicle,
miscellaneous,

    };

};

typedef oscObjectVariable<oscByType *> oscByTypeMember;
typedef oscObjectVariableArray<oscByType *> oscByTypeArrayMember;


}

#endif //OSCBYTYPE_H
