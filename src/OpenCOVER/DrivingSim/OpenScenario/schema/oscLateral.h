/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCLATERAL_H
#define OSCLATERAL_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"

#include "oscVariables.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT Enum_Lateral_purposeType : public oscEnumType
{
public:
static Enum_Lateral_purposeType *instance();
    private:
		Enum_Lateral_purposeType();
	    static Enum_Lateral_purposeType *inst; 
};
class OPENSCENARIOEXPORT oscLateral : public oscObjectBase
{
public:
    oscLateral()
    {
        OSC_ADD_MEMBER(purpose);
    };
    oscEnum purpose;

    enum Enum_Lateral_purpose
    {
position,
steering,

    };

};

typedef oscObjectVariable<oscLateral *> oscLateralMember;


}

#endif //OSCLATERAL_H
