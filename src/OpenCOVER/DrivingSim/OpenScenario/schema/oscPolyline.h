/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCPOLYLINE_H
#define OSCPOLYLINE_H

#include "../oscExport.h"
#include "../oscObjectBase.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscPolyline : public oscObjectBase
{
public:
oscPolyline()
{
    };
        const char *getScope(){return "";};

};

typedef oscObjectVariable<oscPolyline *> oscPolylineMember;
typedef oscObjectVariableArray<oscPolyline *> oscPolylineArrayMember;


}

#endif //OSCPOLYLINE_H
