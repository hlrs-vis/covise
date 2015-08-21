/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#include <oscObjectBase.h>
#include <iostream>


using namespace OpenScenario;

oscFactory<oscObjectBase> oscObjectBase::factory;

oscObjectBase::oscObjectBase()
{
}
oscObjectBase::~oscObjectBase()
{
    
}

void oscObjectBase::initialize(OpenScenarioBase *b)
{
    base = b;
}


int oscObjectBase::parseFromXML(xercesc::DOMElement *currentElement)
{
    return true;
}