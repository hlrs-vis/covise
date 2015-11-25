/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */
#ifndef OSC_TEST_H
#define OSC_TEST_H
#include <oscExport.h>
#include <oscObjectBase.h>
#include <oscObjectVariable.h>
#include <oscDriver.h>
#include <oscCondition.h>
#include <oscPosition.h>

namespace OpenScenario {


/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscTest: public oscObjectBase
{
public:
    oscTest()
    {
        OSC_OBJECT_ADD_MEMBER(driver, "oscDriver");
		OSC_OBJECT_ADD_MEMBER(condition, "oscCondition");
		OSC_OBJECT_ADD_MEMBER(position, "oscPosition");
    };
   oscDriverMember driver;
   oscConditionMember condition;
   oscPositionMember position;

};

typedef oscObjectVariable<oscTest *> oscTestMember;

}

#endif //OSC_TEST_H
