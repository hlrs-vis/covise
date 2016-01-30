/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_ROUTING_H
#define OSC_ROUTING_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"

#include "oscFileHeader.h"
#include "oscObserverTypeB.h"
#include "oscUserDataList.h"


namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscRouting: public oscObjectBase
{
public:
	
    oscRouting()
    {
		OSC_OBJECT_ADD_MEMBER(fileHeader, "oscFileHeader");
		OSC_OBJECT_ADD_MEMBER(observer, "oscObserverTypeB");
		OSC_OBJECT_ADD_MEMBER(userDataList, "oscUserDataList");
    };

	oscFileHeaderMember fileHeader;
	oscObserverTypeBMember observer;
	oscUserDataListArrayMember userDataList;
};

typedef oscObjectVariable<oscRouting *> oscRoutingMember;

}

#endif //OSC_ROUTING_H
