/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#include "oscMember.h"
#include "oscObjectBase.h"
#include "oscFactories.h"
#include "oscVariables.h"

#include <xercesc/dom/DOMDocument.hpp>
#include <xercesc/dom/DOMElement.hpp>


using namespace OpenScenario;


/*****
 * constructor, destructor
 *****/

oscMember::oscMember() :
        value(NULL),
        owner(NULL),
        type(oscMemberValue::MemberTypes::INT),
        parentMember(NULL),
		optional(false)
{

}

oscMember::~oscMember()
{

}



/*****
 * public functions
 *****/

//
void oscMember::registerWith(oscObjectBase* o, short int choiceNumber)
{
    owner = o;
    owner->addMember(this);
	choice = choiceNumber << 1;
}


void oscMember::registerOptionalWith(oscObjectBase *objBase)
{
	optional = true;
    objBase->addMemberToOptional(this);
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

void oscMember::setType(oscMemberValue::MemberTypes t)
{
    type = t;
}

oscMemberValue::MemberTypes oscMember::getType() const
{
    return type;
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


//virtual functions
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

oscMemberValue *oscMember::createValue()
{
	OpenScenario::oscMemberValue *v = oscFactories::instance()->valueFactory->create(type);
	if (type == oscMemberValue::ENUM)
	{
		OpenScenario::oscEnumValue *ev = dynamic_cast<OpenScenario::oscEnumValue *>(v);
		OpenScenario::oscEnum *em = dynamic_cast<OpenScenario::oscEnum *>(this);
		ev->enumType = em->enumType;
	}
	setValue(v);

	return value;
}

oscMemberValue *oscMember::getOrCreateValue()
{
    if (!value)
    {
        createValue();
    }

    return value;
}


//
oscObjectBase *oscMember::getObject() const
{
    return NULL;
}

oscObjectBase *oscMember::getOrCreateObject()
{
    return NULL;
}

oscObjectBase *oscMember::createObject()
{
    return NULL;
}

bool oscMember::exists() const
{
    return false;
}


//
bool oscMember::writeToDOM(xercesc::DOMElement *currentElement, xercesc::DOMDocument *document, bool writeInclude)
{

	if(value != NULL)
	{
		value->writeToDOM(currentElement, document, name.c_str());
	}

	return true;
}

bool oscMember::isChoice()
{
	if (choice >> 1)
	{
		return true;
	}

	return false;
}

int oscMember::choiceNumber()
{
	return choice >> 1;
}

void oscMember::setSelected(short int select)
{
	choice = ((choice >> 1) << 1) | select;
}

bool oscMember::isSelected()
{
	return choice & 1;
}

unsigned char oscMember::validate()
{

	if (choice && !isSelected())
	{
		return oscObjectBase::VAL_valid;
	}


	if (value)
	{
		return oscObjectBase::VAL_valid;
	}
	else 
	{
		oscObjectBase *memberObject = getObject();
		{
			if (!memberObject) 
			{
				if (optional)
				{
					return oscObjectBase::VAL_optional;
				}

				return oscObjectBase::VAL_invalid;
			}

			if (!memberObject->validate())
			{
				return oscObjectBase::VAL_invalid;
			}

			return oscObjectBase::VAL_valid;
		}
	}

}
