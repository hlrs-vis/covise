/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_NAME_REF_ID_USER_DATA_H
#define OSC_NAME_REF_ID_USER_DATA_H

#include "oscExport.h"
#include "oscNameRefId.h"
#include "oscObjectVariable.h"

#include "oscVariables.h"
#include "oscUserDataList.h"


namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscNameRefIdUserData: public oscNameRefId
{
public:
    oscNameRefIdUserData()
    {
        OSC_OBJECT_ADD_MEMBER_OPTIONAL(userDataList, "oscUserDataList");
    };

    oscUserDataListMemberArray userDataList;
};

typedef oscObjectVariable<oscNameRefIdUserData *> oscNameRefIdUserDataMember;

}

#endif /* OSC_NAME_REF_ID_USER_DATA_H */
