/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_FEATURES_H
#define OSC_FEATURES_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectArrayVariable.h"

#include "oscFeature.h"


namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscFeatures: public oscObjectBase
{
public:
    oscFeatures()
    {
        OSC_OBJECT_ADD_MEMBER(feature, "oscFeature");
    };

    oscFeatureMember feature;
};

typedef oscObjectArrayVariable<oscFeatures *> oscFeaturesArrayMember;

}

#endif //OSC_FEATURES_H
