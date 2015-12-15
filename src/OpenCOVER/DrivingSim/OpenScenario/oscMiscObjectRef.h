/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */
#ifndef OSC_MISC_OBJECT_REF_H
#define OSC_MISC_OBJECT_REF_H
#include <oscExport.h>
#include <oscObjectBase.h>
#include <oscObjectVariable.h>
#include <oscVariables.h>
#include <oscNameRefId.h>

namespace OpenScenario {


/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscMiscObjectRef: public oscNameRefId
{
public:
    oscMiscObjectRef()
    {
    };
};

typedef oscObjectVariable<oscMiscObjectRef *> oscMiscObjectRefMember;

}

#endif //OSC_MISC_OBJECT_REF_H
