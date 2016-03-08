/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_MEMBER_H
#define OSC_MEMBER_H

#include "oscExport.h"
#include "oscMemberValue.h"

#include <string>

#include <xercesc/util/XercesDefs.hpp>
XERCES_CPP_NAMESPACE_BEGIN
class DOMDocument;
class DOMElement;
XERCES_CPP_NAMESPACE_END


namespace OpenScenario
{

class oscObjectBase;

/// \class This class represents a generic OpenScenario Member variable
class OPENSCENARIOEXPORT oscMember
{
protected:
    std::string name; ///< name of member
    std::string typeName; ///< type name of member
    oscMemberValue *value;
    oscObjectBase *owner; ///< the parent/owner object of this member
    enum oscMemberValue::MemberTypes type;
    oscMember *parentMember; ///< the parent member of this member

public:
    oscMember(); ///< constructor
    virtual ~oscMember(); ///< destructor

    void registerWith(oscObjectBase *owner);
    void registerChoiceWith(oscObjectBase *objBase);

    void setName(const char *n);
    void setName(std::string &n);
    std::string &getName();
    void setTypeName(const char *tn);
    void setTypeName(std::string &tn);
    std::string getTypeName() const; ///< return the typeName of this member

    virtual void setValue(oscMemberValue *v);
    virtual void setValue(oscObjectBase *t);
    virtual void deleteValue();
    virtual oscMemberValue *getValue();
    virtual oscMemberValue *getGenerateValue();
    void setType(oscMemberValue::MemberTypes t);
    oscMemberValue::MemberTypes getType() const; ///< return the type of this member

    virtual oscObjectBase *getObject() const;
    virtual oscObjectBase *getGenerateObject();
    virtual bool exists() const; ///< for a member of type == oscMemberValue::OBJECT oscObjectVariable...::exists is executed
    oscObjectBase *getOwner() const;
    virtual void setParentMember(oscMember *pm);
    virtual oscMember *getParentMember() const;

    virtual bool writeToDOM(xercesc::DOMElement *currentElement, xercesc::DOMDocument *document);
};


}

#endif //OSC_MEMBER_H
