/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_DISTANCE_LONGITUDINAL_H
#define OSC_DISTANCE_LONGITUDINAL_H

#include <oscExport.h>
#include <oscObjectBase.h>
#include <oscObjectVariable.h>

#include <oscVariables.h>


namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscDistanceLongitudinal: public oscObjectBase
{
public:
    oscDistanceLongitudinal()
    {
        OSC_ADD_MEMBER(refObject);
        OSC_ADD_MEMBER(freespace);
        OSC_ADD_MEMBER(distance);
        OSC_ADD_MEMBER(timeGap);
    };

    oscString refObject;
    oscBool freespace;
    oscDouble distance;
    oscDouble timeGap;
};

typedef oscObjectVariable<oscDistanceLongitudinal *> oscDistanceLongitudinalMember;

}

#endif //OSC_DISTANCE_LONGITUDINAL_H
