/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */
#ifndef OSC_OBSERVER_ID_H
#define OSC_OBSERVER_ID_H
#include <oscExport.h>
#include <oscObjectBase.h>
#include <oscObjectVariable.h>
#include <oscVariables.h>
#include <oscNameRefId.h>
#include <oscCoord.h>

namespace OpenScenario {


/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscObserverId: public oscNameRefId
{
public:
    oscObserverId()
    {
		OSC_ADD_MEMBER(iid);
		OSC_OBJECT_ADD_MEMBER(coord, "oscCoord");
    };
	oscInt iid;
    oscCoordMember coord;
};

typedef oscObjectVariable<oscObserverId *> oscObserverIdMember;

}

#endif //OSC_OBSERVER_ID_H
