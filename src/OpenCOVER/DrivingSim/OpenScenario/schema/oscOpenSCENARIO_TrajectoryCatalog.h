/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCOPENSCENARIO_TRAJECTORYCATALOG_H
#define OSCOPENSCENARIO_TRAJECTORYCATALOG_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"

#include "oscVariables.h"
#include "schema/oscFileHeader.h"
#include "schema/oscTrajectory.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscOpenSCENARIO_TrajectoryCatalog : public oscObjectBase
{
public:
    oscOpenSCENARIO_TrajectoryCatalog()
    {
        OSC_OBJECT_ADD_MEMBER(FileHeader, "oscFileHeader");
        OSC_OBJECT_ADD_MEMBER(Trajectory, "oscTrajectory");
    };
    oscFileHeaderMember FileHeader;
    oscTrajectoryMember Trajectory;

};

typedef oscObjectVariable<oscOpenSCENARIO_TrajectoryCatalog *> oscOpenSCENARIO_TrajectoryCatalogMember;


}

#endif //OSCOPENSCENARIO_TRAJECTORYCATALOG_H
