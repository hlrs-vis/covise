/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */
#ifndef OSC_CHARACTER_APPEARANCE_H
#define OSC_CHARACTER_APPEARANCE_H
#include <oscExport.h>
#include <oscObjectBase.h>
#include <oscObjectVariable.h>
#include <oscVariables.h>

namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscCharacterAppearance: public oscObjectBase
{
public:
    oscCharacterAppearance()
    {
        OSC_ADD_MEMBER(feature);
		OSC_ADD_MEMBER(value);
    };
    oscString feature;
	oscString value;
	
};

typedef oscObjectVariable<oscCharacterAppearance *> oscCharacterAppearanceMember;

}

#endif //OSC_CHARACTER_APPEARANCE_H
