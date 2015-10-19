/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */
#ifndef OSC_OBJECT_BASE_H
#define OSC_OBJECT_BASE_H
#include <oscExport.h>
#include <oscMemberValue.h>
#include <oscMember.h>
#include <oscFactory.h>
#include <string>
#if __cplusplus >= 201103L || defined WIN32
#include <unordered_map>
using std::unordered_map;
#else
#include <tr1/unordered_map>
using std::tr1::unordered_map;
#endif
#include <xercesc/util/XercesDefs.hpp>
XERCES_CPP_NAMESPACE_BEGIN
class DOMDocument;
class DOMElement;
XERCES_CPP_NAMESPACE_END


#define OSC_ADD_MEMBER(varName) varName.setName(#varName); varName.registerWith(this); varName.setType(varName.getValueType());

namespace OpenScenario {

class OpenScenarioBase;

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscObjectBase
{
public:
    typedef unordered_map<std::string, oscMember *> MemberMap;
protected:
    OpenScenarioBase *base;
    MemberMap members; ///< list of all member variables

public:
    oscObjectBase(); ///< constructor
    virtual ~oscObjectBase(); ///< destructor
    virtual void initialize(OpenScenarioBase *b);
    void addMember(oscMember *m);
    OpenScenarioBase *getBase(){return base;};

    bool parseFromXML(xercesc::DOMElement *currentElement);
    bool writeToDOM(xercesc::DOMElement *currentElement, xercesc::DOMDocument *document);

};

}

#endif //OSC_OBJECT_BASE_H
