/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */
#ifndef OSC_DIMENSION_H
#define OSC_DIMENSION_H
#include <oscExport.h>
#include <oscObjectBase.h>
#include <oscObjectVariable.h>
#include <oscVariables.h>

namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscDimension: public oscObjectBase
{
public:
    oscDimension()
    {
		OSC_ADD_MEMBER(front);
		OSC_ADD_MEMBER(rear);
		OSC_ADD_MEMBER(left);
		OSC_ADD_MEMBER(right);
		OSC_ADD_MEMBER(bottom);
		OSC_ADD_MEMBER(top);
    };
    oscDouble front;
	oscDouble rear;
	oscDouble left;
	oscDouble right;
	oscDouble bottom;
	oscDouble top;
};

typedef oscObjectVariable<oscDimension *> oscDimensionMember;

}

#endif //OSC_DIMENSION_H
