/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */
#ifndef OSC_NAMED_OBJECT_H
#define OSC_NAMED_OBJECT_H
#include <oscExport.h>
#include <oscMemberValue.h>
#include <oscObjectBase.h>
#include <oscMember.h>
#include <oscFactory.h>
#include <string>
#include <list>
#include <xercesc/util/XercesDefs.hpp>
XERCES_CPP_NAMESPACE_BEGIN
class DOMDocument;
class DOMElement;
XERCES_CPP_NAMESPACE_END

namespace OpenScenario {

class OpenScenarioBase;

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscNamedObject: public oscObjectBase
{
public:
    oscNamedObject(); ///< constructor
    virtual ~oscNamedObject(); ///< destructor

    virtual int parseFromXML(xercesc::DOMElement *currentElement);

};

}

#endif //OSC_NAMED_OBJECT_H