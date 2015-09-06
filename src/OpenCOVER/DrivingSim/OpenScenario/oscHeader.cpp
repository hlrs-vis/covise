/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#include <oscHeader.h>
#include <xercesc/parsers/XercesDOMParser.hpp>
#include <iostream>


using namespace OpenScenario;

oscHeader::oscHeader(): oscObjectBase()
{
    OSC_ADD_MEMBER(revMajor);
    OSC_ADD_MEMBER(revMinor);
    OSC_ADD_MEMBER(description);
    OSC_ADD_MEMBER(date);
    OSC_ADD_MEMBER(author);
}
oscHeader::~oscHeader()
{
   
}

int oscHeader::parseFromXML(xercesc::DOMElement *currentElement)
{
    return true;
}