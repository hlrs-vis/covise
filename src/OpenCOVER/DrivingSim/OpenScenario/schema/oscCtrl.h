/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCCTRL_H
#define OSCCTRL_H

#include "../oscExport.h"
#include "../oscObjectBase.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"
#include "oscCatalogReference.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscCtrl : public oscObjectBase
{
public:
oscCtrl()
{
        OSC_OBJECT_ADD_MEMBER(CatalogReference, "oscCatalogReference", 0);
    };
    oscCatalogReferenceMember CatalogReference;

};

typedef oscObjectVariable<oscCtrl *> oscCtrlMember;
typedef oscObjectVariableArray<oscCtrl *> oscCtrlArrayMember;


}

#endif //OSCCTRL_H
