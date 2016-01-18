/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_EVENTS_H
#define OSC_EVENTS_H

#include <oscExport.h>
#include <oscObjectBase.h>
#include <oscObjectArrayVariable.h>

#include <oscEvent.h>


namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscEvents: public oscObjectBase
{
public:
    oscEvents()
    {
        OSC_OBJECT_ADD_MEMBER(event, "oscEvent");
    };

    oscEventMember event;
};

typedef oscObjectArrayVariable<oscEvents *> oscEventsArrayMember;

}

#endif /* OSC_EVENTS_H */
