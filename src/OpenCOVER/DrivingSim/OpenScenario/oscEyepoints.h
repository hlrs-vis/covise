/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */
#ifndef OSC_EYEPOINTS_H
#define OSC_EYEPOINTS_H
#include <oscExport.h>
#include <oscObjectBase.h>
#include <oscObjectVariable.h>
#include <oscVariables.h>
#include <oscCoord.h>

namespace OpenScenario {

	/// \class This class represents a generic OpenScenario Object
	class OPENSCENARIOEXPORT oscEyepoints : public oscObjectBase
	{
	public:
		oscEyepoints()
		{
			OSC_ADD_MEMBER(type);
			OSC_OBJECT_ADD_MEMBER(coord, "oscCoord");
		};
		oscString type;
		oscCoordMember coord;
	};

	typedef oscObjectVariable<oscEyepoints *> oscEyepointsMember;

}
#endif //OSC_EYEPOINTS_H