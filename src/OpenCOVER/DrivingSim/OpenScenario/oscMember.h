/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_MEMBER_H
#define OSC_MEMBER_H

#include <oscExport.h>
#include <oscMemberValue.h>
#include <string>
#include <xercesc/util/XercesDefs.hpp>
XERCES_CPP_NAMESPACE_BEGIN
class DOMDocument;
class DOMElement;
XERCES_CPP_NAMESPACE_END

namespace OpenScenario {

class OpenScenarioBase;
class oscObjectBase;


/// \class This class represents a generic OpenScenario Member variable
class OPENSCENARIOEXPORT oscMember
{
protected:

    std::string name; ///< name of member
    std::string typeName; ///< type name of member
    oscMemberValue *value;
    oscObjectBase* owner;
    enum oscMemberValue::MemberTypes type;

public:
    oscMember(); ///< constructor
    void setName(const char *n){name = n;};
    void setName(std::string &n){name = n;};
    void setTypeName(const char *tn) {typeName = tn;};
    void setTypeName(std::string &tn) {typeName = tn;};
    std::string getTypeName() {return typeName;}; ///< return the typeName of this member
    void registerWith(oscObjectBase* owner); ///< constructor
    

    virtual ~oscMember(); ///< destructor
    
    virtual oscMemberValue * getValue() {return value;};
    virtual void setValue(oscMemberValue *v) {value = v;};
    virtual void setValue(oscObjectBase *t){};
    void setType(oscMemberValue::MemberTypes t) {type = t;};
    oscMemberValue::MemberTypes getType() {return type;}; ///< return the type of this member
    virtual bool writeToDOM(xercesc::DOMElement *currentElement, xercesc::DOMDocument *document){if(value!=NULL) value->writeToDOM(currentElement,document,name.c_str());return true;};
    std::string &getName(){return name;};

};


}

#endif //OSC_MEMBER_H
