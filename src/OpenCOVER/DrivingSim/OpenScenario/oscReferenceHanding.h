/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_REFERENCE_HANDING_H
#define OSC_REFERENCE_HANDING_H

#include <oscExport.h>
#include <oscObjectBase.h>
#include <oscObjectVariable.h>

#include <oscVariables.h>


namespace OpenScenario {

class OPENSCENARIOEXPORT conditionRefType: public oscEnumType
{
public:
    static conditionRefType *instance(); 
private:
    conditionRefType();
    static conditionRefType *inst;
};

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscReferenceHanding: public oscObjectBase
{
public:
    oscReferenceHanding()
    {
        OSC_ADD_MEMBER(maneuverId);
        OSC_ADD_MEMBER(eventId);
        OSC_ADD_MEMBER(conditionRef);

        conditionRef.enumType = conditionRefType::instance();
    };

    oscInt maneuverId;
    oscInt eventId;
    oscEnum conditionRef;

    enum conditionRef
    {
        starts,
        ends,
        cancels,
    };
};

typedef oscObjectVariable<oscReferenceHanding *> oscReferenceHandingMember;

}

#endif //OSC_REFERENCE_HANDING_H
