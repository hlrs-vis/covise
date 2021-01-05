/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCSTRINGELEMENT_H
#define OSCSTRINGELEMENT_H

#include "../oscExport.h"
#include "../oscObjectBase.h"

#include "../oscVariables.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscStringElement : public oscObjectBase
{
public:
	oscStringElement()
    {
    };
	std::string value;
	virtual bool parseFromXML(xercesc::DOMElement *currentElement, oscSourceFile *src, bool saveInclude = true) { return true; };
	virtual bool writeToDOM(xercesc::DOMElement *currentElement, xercesc::DOMDocument *document, bool writeInclude = true) { return true; };

};

typedef oscObjectVariable<oscStringElement *> oscStringElementMember;
typedef oscObjectVariableArray<oscStringElement *> oscStringElementArrayMember;


}

#endif //OSCSTRINGELEMENT_H
