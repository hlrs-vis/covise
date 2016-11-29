/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCDESCRIPTION_H
#define OSCDESCRIPTION_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"

#include "oscVariables.h"
#include "schema/oscDescription.h"
#include "schema/oscParameter.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscDescription : public oscObjectBase
{
public:
    oscDescription()
    {
        OSC_ADD_MEMBER(weight);
        OSC_ADD_MEMBER(height);
        OSC_ADD_MEMBER(eyeDistance);
        OSC_ADD_MEMBER(age);
        OSC_ADD_MEMBER(sex);
        OSC_OBJECT_ADD_MEMBER_OPTIONAL(Parameter, "oscParameter");
    };
    oscDouble weight;
    oscDouble height;
    oscDouble eyeDistance;
    oscDouble age;
    oscEnum sex;
    oscParameterMember Parameter;

};

typedef oscObjectVariable<oscDescription *> oscDescriptionMember;


}

#endif //OSCDESCRIPTION_H
