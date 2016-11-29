/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCSELECTION_H
#define OSCSELECTION_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"

#include "oscVariables.h"
#include "schema/oscMembers.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscSelection : public oscObjectBase
{
public:
    oscSelection()
    {
        OSC_ADD_MEMBER(name);
        OSC_OBJECT_ADD_MEMBER(Members, "oscMembers");
    };
    oscString name;
    oscMembersMember Members;

};

typedef oscObjectVariable<oscSelection *> oscSelectionMember;


}

#endif //OSCSELECTION_H
