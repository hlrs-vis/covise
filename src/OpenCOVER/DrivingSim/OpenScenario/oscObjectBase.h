/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */
#ifndef OSC_OBJECT_BASE_H
#define OSC_OBJECT_BASE_H
#include <oscExport.h>
#include <oscMemberValue.h>
#include <oscMember.h>

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


//varName is of type oscMember
//set the name of varName as string of varName (oscMember[.cpp,.h])
//register varName in MemberMap members ([oscMember,oscObjectBase][.cpp,.h])
//set the type of variable varName ([oscMember,oscMemberValue,oscVariables][.cpp,.h])
//set the type name of varName (element name in xml file) (oscMember[.cpp,.h])
#define OSC_ADD_MEMBER(varName) varName.setName(#varName); varName.registerWith(this); varName.setType(varName.getValueType())
#define OSC_OBJECT_ADD_MEMBER(varName,typeName) varName.setName(#varName); varName.registerWith(this); varName.setType(varName.getValueType()); varName.setTypeName(typeName)

namespace OpenScenario {

class OpenScenarioBase;
class oscSourceFile;

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscObjectBase
{
public:
    typedef unordered_map<std::string, oscMember *> MemberMap;

protected:
    OpenScenarioBase *base;
    MemberMap members; ///< list of all member variables
    oscSourceFile *source;

public:
    oscObjectBase(); ///< constructor
    virtual ~oscObjectBase(); ///< destructor

    virtual void initialize(OpenScenarioBase *b, oscSourceFile *s);
    void addMember(oscMember *m);
    OpenScenarioBase *getBase();
    oscSourceFile *getSource() const;

    bool parseFromXML(xercesc::DOMElement *currentElement, oscSourceFile *src);
    bool writeToDOM(xercesc::DOMElement *currentElement, xercesc::DOMDocument *document);

};

}

#endif //OSC_OBJECT_BASE_H
