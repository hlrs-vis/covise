/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */
#ifndef OSC_LIGHTING_H
#define OSC_LIGHTING_H
#include <oscExport.h>
#include <oscObjectBase.h>
#include <oscObjectVariable.h>
#include <oscVariables.h>
#include <oscCoord.h>
#include <oscLight.h>

namespace OpenScenario {

	/// \class This class represents a generic OpenScenario Object
	class OPENSCENARIOEXPORT oscLighting : public oscObjectBase
	{
	public:
		oscLighting()
		{
			OSC_ADD_MEMBER(type);
			OSC_OBJECT_ADD_MEMBER(coord, "oscCoord");
			OSC_OBJECT_ADD_MEMBER(light, "oscLight");
			OSC_ADD_MEMBER(frequency);
		};
		oscString type;
		oscCoordMember coord;
		oscLightMember light;
		oscDouble frequency;
	};

	typedef oscObjectVariable<oscLighting *> oscLightingMember;

}
#endif //OSC_LIGHTING_H