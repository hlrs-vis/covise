/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCSUN_H
#define OSCSUN_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"
#include "oscObjectVariableArray.h"

#include "oscVariables.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscSun : public oscObjectBase
{
public:
oscSun()
{
        OSC_ADD_MEMBER(intensity);
        OSC_ADD_MEMBER(azimuth);
        OSC_ADD_MEMBER(elevation);
    };
    oscDouble intensity;
    oscDouble azimuth;
    oscDouble elevation;

};

typedef oscObjectVariable<oscSun *> oscSunMember;
typedef oscObjectVariableArray<oscSun *> oscSunArrayMember;


}

#endif //OSCSUN_H
