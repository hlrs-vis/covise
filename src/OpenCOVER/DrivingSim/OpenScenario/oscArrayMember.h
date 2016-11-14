/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_MEMBER_ARRAY_H
#define OSC_MEMBER_ARRAY_H

#include "oscExport.h"
#include "oscMember.h"

#include <vector>

#include <xercesc/util/XercesDefs.hpp>
XERCES_CPP_NAMESPACE_BEGIN
class DOMDocument;
class DOMElement;
XERCES_CPP_NAMESPACE_END


namespace OpenScenario
{

/// \class This class represents a Member variable storing an array of one kind of values
class OPENSCENARIOEXPORT oscArrayMember: public oscMember, public std::vector<oscObjectBase *>
{
public:
    oscArrayMember(); ///< constructor
    virtual ~oscArrayMember(); ///< destructor

    virtual xercesc::DOMElement *writeArrayMemberToDOM(xercesc::DOMElement *currentElement, xercesc::DOMDocument *document);

	int findObjectIndex(oscObjectBase *object);
};

}

#endif //OSC_MEMBER_ARRAY_H
