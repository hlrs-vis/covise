/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_PARTNER_OBJECT_H
#define OSC_PARTNER_OBJECT_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"

#include "oscVariables.h"


namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscPartnerObject: public oscObjectBase
{
public:
    oscPartnerObject()
    {
        OSC_ADD_MEMBER(object);
    };

    oscString object;
};

typedef oscObjectVariable<oscPartnerObject *> oscPartnerObjectMember;

}

#endif /* OSC_PARTNER_OBJECT_H */
