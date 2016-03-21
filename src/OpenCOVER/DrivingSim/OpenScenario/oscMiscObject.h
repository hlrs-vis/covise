/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_MISC_OBJECT_H
#define OSC_MISC_OBJECT_H

#include "oscExport.h"
#include "oscNameRefId.h"
#include "oscObjectVariable.h"

#include "oscVariables.h"
#include "oscFileHeader.h"
#include "oscDimensionTypeA.h"
#include "oscFile.h"


namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscMiscObject: public oscNameRefId
{
public:
    oscMiscObject()
    {
        OSC_OBJECT_ADD_MEMBER(fileHeader, "oscFileHeader");
        OSC_ADD_MEMBER(type);
        OSC_ADD_MEMBER(mass);
        OSC_OBJECT_ADD_MEMBER(dimension, "oscDimensionTypeA");
        OSC_OBJECT_ADD_MEMBER(geometry, "oscFile");
    };
    oscFileHeaderMember fileHeader;
    oscString type;
    oscDouble mass;
    oscDimensionTypeAMember dimension;
    oscFileMember geometry;
};

typedef oscObjectVariable<oscMiscObject *> oscMiscObjectMember;

}

#endif //OSC_MISC_OBJECT_H
