/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_NAME_USER_DATA_H
#define OSC_NAME_USER_DATA_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"

#include "oscVariables.h"
#include "oscUserDataList.h"


namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscNameUserData: public oscObjectBase
{
public:
    oscNameUserData()
    {
        OSC_ADD_MEMBER(name);
        OSC_OBJECT_ADD_MEMBER_OPTIONAL(userDataList, "oscUserDataList");
    }

    oscString name;
    oscUserDataListMemberArray userDataList;
};

typedef oscObjectVariable<oscNameUserData *> oscNameUserDataMember;

}

#endif //OSC_NAME_USER_DATA_H
