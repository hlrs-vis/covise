/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */
#ifndef OSC_ACCELERATION_H
#define OSC_ACCELERATION_H
#include <oscExport.h>
#include <oscObjectBase.h>
#include <oscObjectVariable.h>
#include <oscVariables.h>
#include <oscObjectRef.h>

namespace OpenScenario {


	/// \class This class represents a generic OpenScenario Object
	class OPENSCENARIOEXPORT oscAcceleration : public oscObjectBase
	{
	public:
		oscAcceleration()
		{
			OSC_OBJECT_ADD_MEMBER(objectRef, "oscObjectRef");
		};
		oscObjectRefMember objectRef;
	};

	typedef oscObjectVariable<oscAcceleration *> oscAccelerationMember;

}

#endif //OSC_ACCELERATION_H
