/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_NOTIFY_H
#define OSC_NOTIFY_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"

#include "oscVariables.h"


namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscNotify: public oscObjectBase
{
public:
    oscNotify()
    {
        OSC_ADD_MEMBER(text);
    };

    oscString text;
};

typedef oscObjectVariable<oscNotify *> oscNotifyMember;

}

#endif //OSC_NOTIFY_H
