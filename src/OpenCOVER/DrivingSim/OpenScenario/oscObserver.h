/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */
#ifndef OSC_OBSERVER_H
#define OSC_OBSERVER_H
#include <oscExport.h>
#include <oscObjectBase.h>
#include <oscObjectVariable.h>
#include <oscVariables.h>
#include <oscHeader.h>
#include <oscNamedObject.h>
#include <oscFrustum.h>
#include <oscFilter.h>

namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscObserver: public oscNamedObject
{
public:
    oscObserver()
    {
        OSC_OBJECT_ADD_MEMBER(header,"oscHeader");
		OSC_ADD_MEMBER(refId);
		OSC_ADD_MEMBER(type);
		OSC_OBJECT_ADD_MEMBER(frustum,"oscFrustum");
		OSC_OBJECT_ADD_MEMBER(filter,"oscFilter");
    };
    oscHeaderMember header;
	oscInt refId;
	oscString type;
	oscFrustumMember frustum;
	oscFilterMember filter;
};

typedef oscObjectVariable<oscObserver *> oscObserverMember;

}

#endif //OSC_OBSERVER_H
