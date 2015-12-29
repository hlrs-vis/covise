/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */
#ifndef OSC_GENERAL_H
#define OSC_GENERAL_H
#include <oscExport.h>
#include <oscObjectBase.h>
#include <oscObjectVariable.h>
#include <oscVariables.h>
#include <oscNamedObject.h>

namespace OpenScenario {


/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscGeneral: public oscNamedObject
{
public:
    oscGeneral()
    {
		OSC_ADD_MEMBER(closed);
    };	
	oscBool closed;
	
};

typedef oscObjectVariable<oscGeneral *> oscGeneralMember;

}

#endif //OSC_GENERAL_H