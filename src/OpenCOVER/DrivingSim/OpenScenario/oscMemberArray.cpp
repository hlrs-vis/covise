/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#include "oscMemberArray.h"
#include "oscObjectBase.h"

#include <xercesc/dom/DOMDocument.hpp>
#include <xercesc/dom/DOMElement.hpp>
#include <xercesc/util/XMLString.hpp>


using namespace OpenScenario;


oscMemberArray::oscMemberArray(): oscMember(), std::vector<oscObjectBase *>()
{

}

oscMemberArray::~oscMemberArray()
{

}


//
xercesc::DOMElement *oscMemberArray::writeMemberArrayToDOM(xercesc::DOMElement *currentElement, xercesc::DOMDocument *document)
{
    xercesc::DOMElement *aMemberElement = document->createElement(xercesc::XMLString::transcode(name.c_str()));
    currentElement->appendChild(aMemberElement);

    return aMemberElement;
}
