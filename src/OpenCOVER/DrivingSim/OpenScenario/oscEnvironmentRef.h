/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */
#ifndef OSC_ENVIRONMENT_REF_H
#define OSC_ENVIRONMENT_REF_H
#include <oscExport.h>
#include <oscObjectBase.h>
#include <oscObjectVariable.h>
#include <oscVariables.h>
#include <oscCatalogRef.h>
#include <oscUserData.h>
#include <oscFile.h>

namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscEnvironmentRef: public oscObjectBase
{
    
public:
    oscEnvironmentRef()
    {
        OSC_OBJECT_ADD_MEMBER(catalogRef,"oscCatalogRef");
		OSC_OBJECT_ADD_MEMBER(userData,"oscUserData");
	    OSC_OBJECT_ADD_MEMBER(include,"oscFile");
    };
    oscCatalogRefMember catalogRef;
	oscUserDataMember userData;
    oscFileMember include;
};

typedef oscObjectVariable<oscEnvironmentRef *> oscEnvironmentRefMember;

}

#endif //OSC_ENVIRONMENT_REF_H
