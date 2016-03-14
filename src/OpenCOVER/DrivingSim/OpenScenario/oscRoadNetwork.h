/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_ROAD_NETWORK_H
#define OSC_ROAD_NETWORK_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"

#include "oscFile.h"
#include "oscUserDataList.h"


namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscRoadNetwork: public oscObjectBase
{
public:
    oscRoadNetwork()
    {
        OSC_OBJECT_ADD_MEMBER(openDrive, "oscFile");
        OSC_OBJECT_ADD_MEMBER(sceneGraph, "oscFile");
        OSC_OBJECT_ADD_MEMBER_OPTIONAL(userDataList, "oscUserDataList");
    };

    oscFileMember openDrive;
    oscFileMember sceneGraph;
    oscUserDataListMemberArray userDataList;
};

typedef oscObjectVariable<oscRoadNetwork *> oscRoadNetworkMember;

}

#endif //OSC_ROAD_NETWORK_H
