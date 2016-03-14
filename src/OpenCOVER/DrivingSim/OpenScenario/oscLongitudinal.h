/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_LONGITUDINAL_H
#define OSC_LONGITUDINAL_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"

#include "oscTiming.h"


namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscLongitudinal: public oscObjectBase
{
public:
    oscLongitudinal()
    {
        OSC_OBJECT_ADD_MEMBER_OPTIONAL(timing, "oscTiming");
    };

    oscTimingMember timing;
};

typedef oscObjectVariable<oscLongitudinal *> oscLongitudinalMember;

}

#endif /* OSC_LONGITUDINAL_H */
