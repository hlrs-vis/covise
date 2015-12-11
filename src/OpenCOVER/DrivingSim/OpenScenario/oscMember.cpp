/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#include <oscMember.h>
#include <oscObjectBase.h>

using namespace OpenScenario;


/*oscMember::oscMember(std::string &n, oscMemberValue::MemberTypes t, oscObjectBase* owner, oscMemberValue *&mv): value(mv)
{
    name = n;
    type = t;
    owner->addMember(this);
}*/

oscMember::oscMember()
{
    value = NULL;
    owner = NULL;
    type = oscMemberValue::MemberTypes::INT;
}

oscMember::~oscMember()
{
}


void oscMember::registerWith(oscObjectBase* o)
{
    owner = o;
    owner->addMember(this);
}

// set, get of name and typeName
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

std::string oscMember::getTypeName()
{
    return typeName;
}

//set, get of value and type
//
void oscMember::setValue(oscMemberValue *v)
{
    value = v;
}

void oscMember::setValue(oscObjectBase *t)
{

}

oscMemberValue *oscMember::getValue()
{
    return value;
}

void oscMember::setType(oscMemberValue::MemberTypes t)
{
    type = t;
}

oscMemberValue::MemberTypes oscMember::getType()
{
    return type;
}


//
const oscObjectBase *oscMember::getObject()
{
    return NULL;
}

bool oscMember::exists()
{
    return false;
}

oscObjectBase *oscMember::getOwner()
{
    return owner;
}


//
bool oscMember::writeToDOM(xercesc::DOMElement *currentElement, xercesc::DOMDocument *document)
{
    if(value != NULL)
    {
        value->writeToDOM(currentElement,document,name.c_str());
    }

    return true;
}
