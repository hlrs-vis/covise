/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */
#ifndef OSC_ENTITIES_H
#define OSC_ENTITIES_H
#include <oscExport.h>
#include <oscObjectBase.h>
#include <oscObjectVariable.h>
#include <oscObject.h>
#include <oscUserData.h>
#include <oscFile.h>


namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscEntities: public oscObjectBase
{
public:
    oscEntities()
    {
       OSC_OBJECT_ADD_MEMBER(object,"oscObject");
       OSC_OBJECT_ADD_MEMBER(userData,"oscUserData");
	   OSC_OBJECT_ADD_MEMBER(include,"oscFile");

    };
    oscObjectMember object;
    oscUserDataMember userData;
    oscFileMember include;
};

typedef oscObjectVariable<oscEntities *> oscEntitiesMember;

}

#endif //OSC_ENTITIES_H
