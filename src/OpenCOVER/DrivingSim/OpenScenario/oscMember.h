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

class oscObjectBase;

/// \class This class represents a generic OpenScenario Member variable
class OPENSCENARIOEXPORT oscMember
{
protected:

    std::string name; ///< name of member
    std::string typeName; ///< type name of member
    oscMemberValue *value;
    oscObjectBase *owner;
    enum oscMemberValue::MemberTypes type;

public:
    oscMember(); ///< constructor
    virtual ~oscMember(); ///< destructor

    void registerWith(oscObjectBase *owner);

    void setName(const char *n){name = n;};
    void setName(std::string &n){name = n;};
    std::string &getName(){return name;};
    void setTypeName(const char *tn) {typeName = tn;};
    void setTypeName(std::string &tn) {typeName = tn;};
    std::string getTypeName() {return typeName;}; ///< return the typeName of this member

    virtual void setValue(oscMemberValue *v) {value = v;};
    virtual void setValue(oscObjectBase *t){};
    virtual oscMemberValue *getValue() {return value;};
    void setType(oscMemberValue::MemberTypes t) {type = t;};
    oscMemberValue::MemberTypes getType() {return type;}; ///< return the type of this member

    virtual const oscObjectBase *getObject(){return NULL;}
    virtual bool exists(){return false;}; ///<for a member of type == oscMemberValue::OBJECT oscObjectVariable::exists is executed
    oscObjectBase *getOwner() {return owner;};

    virtual bool writeToDOM(xercesc::DOMElement *currentElement, xercesc::DOMDocument *document){if(value!=NULL) value->writeToDOM(currentElement,document,name.c_str());return true;};

};


}

#endif //OSC_MEMBER_H
