/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#include "oscMember.h"
#include "oscObjectBase.h"
#include "oscFactories.h"

#include <xercesc/dom/DOMDocument.hpp>
#include <xercesc/dom/DOMElement.hpp>


using namespace OpenScenario;


oscMember::oscMember()
{
    value = NULL;
    owner = NULL;
    type = oscMemberValue::MemberTypes::INT;
    parentMember = NULL;
}

oscMember::~oscMember()
{
}


void oscMember::registerWith(oscObjectBase* o)
{
    owner = o;
    owner->addMember(this);
}


//
void oscMember::setName(const char *n)
{
    name = n;
}

void oscMember::setName(std::string &n)
{
    name = n;
}

std::string &oscMember::getName()
{
    return name;
}

void oscMember::setTypeName(const char *tn)
{
    typeName = tn;
}

void oscMember::setTypeName(std::string &tn)
{
    typeName = tn;
}

std::string oscMember::getTypeName() const
{
    return typeName;
}


//
void oscMember::setValue(oscMemberValue *v)
{
    value = v;
}

void oscMember::setValue(oscObjectBase *t)
{

}

void oscMember::deleteValue()
{

}

oscMemberValue *oscMember::getValue()
{
	return value;
}

oscMemberValue *oscMember::getGenerateValue()
{
    if (!value)
    {
        OpenScenario::oscMemberValue *v = oscFactories::instance()->valueFactory->create(type);
        setValue(v);
    }

    return value;
}

void oscMember::setType(oscMemberValue::MemberTypes t)
{
    type = t;
}

oscMemberValue::MemberTypes oscMember::getType() const
{
    return type;
}


//
oscObjectBase *oscMember::getObject() const
{
    return NULL;
}

oscObjectBase *oscMember::getGenerateObject()
{
    return NULL;
}

bool oscMember::exists() const
{
    return false;
}

oscObjectBase *oscMember::getOwner() const
{
    return owner;
}

void oscMember::setParentMember(oscMember *pm)
{
    parentMember = pm;
}

oscMember *oscMember::getParentMember() const
{
    return parentMember;
}


//
bool oscMember::writeToDOM(xercesc::DOMElement *currentElement, xercesc::DOMDocument *document)
{
    if(value != NULL)
    {
        value->writeToDOM(currentElement, document, name.c_str());
    }

    return true;
}
