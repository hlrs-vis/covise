/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */
#ifndef OSC_OBJECT_REF_H
#define OSC_OBJECT_REF_H
#include <oscExport.h>
#include <oscObjectBase.h>
#include <oscObjectVariable.h>
#include <oscVariables.h>
#include <oscObject.h>

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
class OPENSCENARIOEXPORT oscObjectRef: public oscObjectBase
{
public:
	
    enum reference
    {
        relative,
		absolute,
    };
    oscObjectRef()
    {
		OSC_OBJECT_ADD_MEMBER(object, "oscObject");
		OSC_ADD_MEMBER(reference);
		reference.enumType = referenceType::instance();
    };
	
	oscObjectMember object;
	oscEnum reference;
};

typedef oscObjectVariable<oscObjectRef *> oscObjectRefMember;

}

#endif //OSC_OBJECT_REF_H
