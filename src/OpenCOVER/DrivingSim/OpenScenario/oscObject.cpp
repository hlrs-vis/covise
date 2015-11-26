/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */
#include <oscObject.h>

using namespace OpenScenario;

conditionType::conditionType()
{
    addEnum("exceed",oscObject::exceed);
    addEnum("deceed",oscObject::deceed );
}

conditionType *conditionType::instance()
{
    if(inst == NULL) 
        inst = new conditionType(); 
    return inst;
}

conditionType *conditionType::inst=NULL;