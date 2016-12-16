/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCUSERDATALIST_H
#define OSCUSERDATALIST_H

#include "../oscExport.h"
#include "../oscObjectBase.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"
#include "oscUserData.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscUserDataList : public oscObjectBase
{
public:
oscUserDataList()
{
        OSC_OBJECT_ADD_MEMBER_OPTIONAL(UserData, "oscUserData");
    };
    oscUserDataArrayMember UserData;

};

typedef oscObjectVariable<oscUserDataList *> oscUserDataListMember;
typedef oscObjectVariableArray<oscUserDataList *> oscUserDataListArrayMember;


}

#endif //OSCUSERDATALIST_H
