/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_FEATURE_H
#define OSC_FEATURE_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"

#include "oscVariables.h"


namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscFeature: public oscObjectBase
{
public:
    oscFeature()
    {
        OSC_ADD_MEMBER(type);
        OSC_ADD_MEMBER(availability);
    };

    oscString type;
    oscBool availability;
};

typedef oscObjectVariable<oscFeature *> oscFeatureMember;

}

#endif //OSC_FEATURES_H
