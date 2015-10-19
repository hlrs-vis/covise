/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#include <oscNamedObject.h>
#include <xercesc/parsers/XercesDOMParser.hpp>
#include <iostream>


using namespace OpenScenario;

oscNamedObject::oscNamedObject(): oscObjectBase()
{
		OSC_ADD_MEMBER(name);
		OSC_ADD_MEMBER(userData);
		OSC_ADD_MEMBER(include);
}
oscNamedObject::~oscNamedObject()
{
    
}