/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */
#ifndef OSC_PEDESTRIAN_H
#define OSC_PEDESTRIAN_H
#include <oscExport.h>
#include <oscFile.h>
#include <oscObjectBase.h>
#include <oscObjectVariable.h>
#include <oscVariables.h>
#include <oscHeader.h>
#include <oscDimension.h>
#include <oscBehavior.h>

namespace OpenScenario {

class OpenScenarioBase;

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscPedestrian: public oscObjectBase
{
public:
    oscPedestrian()
    {
        OSC_ADD_MEMBER(header);
        OSC_ADD_MEMBER(name);
		OSC_ADD_MEMBER(refId);
        OSC_ADD_MEMBER(model);
		OSC_ADD_MEMBER(mass);
        OSC_ADD_MEMBER(behavior);
		OSC_ADD_MEMBER(demension);
        OSC_ADD_MEMBER(Geometry);
    };
    oscHeaderMember header;
    oscInt name;
	oscInt refId;
	oscString model;
	oscDouble mass;
	oscBehaviorMember behavior;
	oscDimensionMember demension;
	oscFileMember Geometry;
};

typedef oscObjectVariable<oscPedestrian *> oscPedestrianMember;

}

#endif //OSC_PEDESTRIAN_H