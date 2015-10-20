/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */
#ifndef OSC_CATALOGREF_H
#define OSC_CATALOGREF_H
#include <oscExport.h>
#include <oscObjectBase.h>
#include <oscObjectVariable.h>
#include <oscVariables.h>

namespace OpenScenario {

class OpenScenarioBase;

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscCatalogRef: public oscObjectBase
{
public:
    oscCatalogRef()
    {
        OSC_ADD_MEMBER(name);
        OSC_ADD_MEMBER(refId);
    };
    oscString catalogId;
    oscInt refId;
};

typedef oscObjectVariable<oscCatalogRef *> oscCatalogRefMember;

}

#endif //OSC_CATALOGREF_H