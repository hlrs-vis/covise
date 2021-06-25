/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCSETENVIRONMENT_H
#define OSCSETENVIRONMENT_H

#include "../oscExport.h"
#include "../oscObjectBase.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"
#include "oscEnvironment.h"
#include "oscCatalogReference.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscSetEnvironment : public oscObjectBase
{
public:
oscSetEnvironment()
{
        OSC_OBJECT_ADD_MEMBER(Environment, "oscEnvironment", 1);
        OSC_OBJECT_ADD_MEMBER(CatalogReference, "oscCatalogReference", 1);
    };
        const char *getScope(){return "/OSCGlobalAction";};
    oscEnvironmentMember Environment;
    oscCatalogReferenceMember CatalogReference;

};

typedef oscObjectVariable<oscSetEnvironment *> oscSetEnvironmentMember;
typedef oscObjectVariableArray<oscSetEnvironment *> oscSetEnvironmentArrayMember;


}

#endif //OSCSETENVIRONMENT_H
