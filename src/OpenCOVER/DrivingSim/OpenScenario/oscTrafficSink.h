/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_TRAFFIC_SINK_H
#define OSC_TRAFFIC_SINK_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"

#include "oscPosition.h"


namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscTrafficSink: public oscObjectBase
{
public:
    oscTrafficSink()
    {	
        OSC_OBJECT_ADD_MEMBER(position, "oscPosition");
    };

    oscPositionMember position;
};

typedef oscObjectVariable<oscTrafficSink *> oscTrafficSinkMember;

}

#endif //OSC_TRAFFIC_SINK_H
