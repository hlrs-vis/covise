/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_POLYLINE_H
#define OSC_POLYLINE_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"


namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscPolyline: public oscObjectBase
{
public:

    oscPolyline()
    {

    };

};

typedef oscObjectVariable<oscPolyline *> oscPolylineMember;

}

#endif /* OSC_POLYLINE_H */
