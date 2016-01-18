/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_OBJECT_BASE_H
#define OSC_OBJECT_BASE_H

#include <oscExport.h>
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
    MemberMap members; ///< list of all member variables
    OpenScenarioBase *base;
    oscSourceFile *source;
    oscObjectBase *parentObj; ///< the parent of this objectBase
    oscMember *ownMem; ///< the member which store this objectBase as a valueT in oscObjectVariable or oscObjectArrayVariable

    const std::string XMLNS = "xmlns"; ///< define a namespace
    const std::string XMLNSOSC = "xmlns:osc"; ///< namespace with name osc
    const std::string XMLBASE = "xml:base"; ///< filename from which the element and its children were read
    const std::string INCLUDE = "osc:include"; ///< element name for include
    const std::string INCLNS = "http://www.w3.org/2001/XInclude"; /*!< XInclude defines a namespace associated with this URI
                                                                       it is no link, it is treated as a normal string (as a formal identifier)*/

public:
    oscObjectBase(); ///< constructor
    virtual ~oscObjectBase(); ///< destructor

    virtual void initialize(OpenScenarioBase *b, oscObjectBase *parentObject, oscMember *ownMember, oscSourceFile *s); ///< params: base, parentObj, ownMem, source
    void addMember(oscMember *m);
    void setBase(OpenScenarioBase *b);
    void setSource(oscSourceFile *s);
    MemberMap getMembers() const;
    OpenScenarioBase *getBase() const;
    oscSourceFile *getSource() const;

    void setParentObj(OpenScenarioBase *pObj);
    void setOwnMember(oscMember *om);
    oscObjectBase *getParentObj() const;
    oscMember *getOwnMember() const;

    bool parseFromXML(xercesc::DOMElement *currentElement, oscSourceFile *src);
    bool writeToDOM(xercesc::DOMElement *currentElement, xercesc::DOMDocument *document);

private:
    void addXInclude(xercesc::DOMElement *currElem, xercesc::DOMDocument *doc, const XMLCh *fileHref); ///< during write adds the include node
    oscSourceFile *determineSrcFile(xercesc::DOMElement *memElem, oscSourceFile *srcF); ///< determine which source file to use
};

}

#endif //OSC_OBJECT_BASE_H
