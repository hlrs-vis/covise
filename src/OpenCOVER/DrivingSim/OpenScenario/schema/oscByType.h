/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCBYTYPE_H
#define OSCBYTYPE_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"

#include "oscVariables.h"
#include "schema/oscByType.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscByType : public oscObjectBase
{
public:
    oscByType()
    {
        OSC_ADD_MEMBER(type);
    };
    oscEnum type;

};

typedef oscObjectVariable<oscByType *> oscByTypeMember;


}

#endif //OSCBYTYPE_H
