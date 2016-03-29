/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_OBJECT_VARIABLE_ARRAY_H
#define OSC_OBJECT_VARIABLE_ARRAY_H

#include "oscExport.h"
#include "oscMemberArray.h"
#include "oscObjectVariableBase.h"


namespace OpenScenario
{

template<typename T>
class OPENSCENARIOEXPORT oscObjectVariableArray : public oscObjectVariableBase<T, oscMemberArray>
{

};

}

#endif //OSC_OBJECT_VARIABLE_ARRAY_H
