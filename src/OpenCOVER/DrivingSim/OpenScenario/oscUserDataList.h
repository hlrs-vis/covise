/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_USERDATA_LIST_H
#define OSC_USERDATA_LIST_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariableArray.h"

#include "oscUserData.h"


namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscUserDataList: public oscObjectBase
{
public:
    oscUserDataList()
    {
        OSC_OBJECT_ADD_MEMBER(userData, "oscUserData");
    };

    oscUserDataMember userData;
};

typedef oscObjectVariableArray<oscUserDataList *> oscUserDataListMemberArray;

}

#endif /* OSC_USERDATA_LIST_H */
