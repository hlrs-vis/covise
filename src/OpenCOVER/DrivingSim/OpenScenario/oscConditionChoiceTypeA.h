/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_CONDITION_CHOICE_TYPE_A_H
#define OSC_CONDITION_CHOICE_TYPE_A_H

#include <oscExport.h>
#include <oscConditionChoiceBase.h>
#include <oscObjectVariable.h>

#include <oscVariables.h>


namespace OpenScenario {

class OPENSCENARIOEXPORT referenceType: public oscEnumType
{
public:
    static referenceType *instance();
private:
    referenceType();
    static referenceType *inst;
};

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscConditionChoiceTypeA: public oscConditionChoiceBase
{
public:
    oscConditionChoiceTypeA()
    {
        OSC_ADD_MEMBER(reference);

        reference.enumType = referenceType::instance();
    };

    oscEnum reference;

    enum reference
    {
        relative,
        absolute,
    };
};

typedef oscObjectVariable<oscConditionChoiceTypeA *> oscConditionChoiceTypeAMember;

}

#endif //OSC_CONDITION_CHOICE_TYPE_A_H
