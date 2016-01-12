/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   21.05.2010
**
**************************************************************************/

#include "osccommands.hpp"

#include "src/data/oscsystem/oscbase.hpp"
#include "src/data/oscsystem/oscelement.hpp"

// OpenScenario //
#include "OpenScenarioBase.h"
#include "oscObjectBase.h"
#include "oscObject.h"
#include "oscMember.h"
#include "oscMemberValue.h"
#include "oscVariables.h"
#include "oscCatalogs.h"

using namespace OpenScenario;

//#########################//
// AddOSCObjectCommand //
//#########################//

AddOSCObjectCommand::AddOSCObjectCommand(OpenScenario::oscObjectBase *parentObject, OSCBase *base, const std::string &name, OSCElement *element, DataCommand *parent)
    : DataCommand(parent)
	, element_(element)
	, parentObject_(parentObject)
	, oscBase_(base)
{
    // Check for validity //
    //
	OpenScenario::oscObjectBase::MemberMap members = parentObject_->getMembers();
	member_ = members[name];
    if ((name == "") || !member_)
    {
        setInvalid(); // Invalid
        setText(QObject::tr("AddOSCObjectCommand: Internal error! No name specified or member already present."));
        return;
    }
    else
    {
        setValid();
        setText(QObject::tr("AddOSCObject"));
    }

	typeName_ = member_->getTypeName();

	openScenarioBase_ = parentObject_->getBase();
}

/*! \brief .
*
*/
AddOSCObjectCommand::~AddOSCObjectCommand()
{
    // Clean up //
    //
    if (isUndone())
    {
//        delete object_;
    }
    else
    {
        // nothing to be done (object is now owned by the road)
    }
}

/*! \brief .
*
*/
void
AddOSCObjectCommand::redo()
{
	OpenScenario::oscObjectBase *obj = oscFactories::instance()->objectFactory->create(typeName_);

	if(obj)
	{
		obj->initialize(openScenarioBase_, parentObject_, member_, NULL);	
		member_->setValue(obj);


		element_->setObjectBase(obj);
		oscBase_->addOSCElement(element_);
	}

	setRedone();
}

/*! \brief
*
*/
void
AddOSCObjectCommand::undo()
{
	const OpenScenario::oscObjectBase *obj = member_->getObject();
	member_->deleteValue();

	element_->setObjectBase(NULL);
	oscBase_->delOSCElement(element_);

   setUndone();
}

//#########################//
// RemoveOSCObjectCommand //
//#########################//

RemoveOSCObjectCommand::RemoveOSCObjectCommand(OSCElement *element, DataCommand *parent) // or oscObjectBase ??
    : DataCommand(parent)
	, element_(element)
{
    // Check for validity //
    //
    if (!element_)
    {
        setInvalid(); // Invalid
        setText(QObject::tr("RemoveOSCObjectCommand: Internal error! No element specified."));
        return;
    }
    else
    {
        setValid();
        setText(QObject::tr("RemoveOSCObject"));
    }
	oscBase_ = element_->getOSCBase();
	openScenarioBase_ = oscBase_->getOpenScenarioBase();
	object_ = element_->getObject();
	parentMember_ = object_->getOwnMember();
}

/*! \brief .
*
*/
RemoveOSCObjectCommand::~RemoveOSCObjectCommand()
{
    // Clean up //
    //
    if (isUndone())
    {
        // nothing to be done (object is now owned by the road)
    }
    else
    {
//        delete object_;
    }
}

/*! \brief .
*
*/
void
RemoveOSCObjectCommand::redo()
{
    element_->setObjectBase(NULL);				// todo: delete OpenScenario object/member
	oscBase_->delOSCElement(element_);

	if (parentMember_)
	{
		parentMember_->deleteValue();
	}

    setRedone();
}

/*! \brief
*
*/
void
RemoveOSCObjectCommand::undo()
{
    element_->setObjectBase(object_);
	oscBase_->addOSCElement(element_);

	object_->setOwnMember(parentMember_);

    setUndone();
}

//#########################//
// AddOSCEnumValueCommand //
//#########################//

AddOSCEnumValueCommand::AddOSCEnumValueCommand(const OpenScenario::oscObjectBase *parentObject, const std::string &name, int value, DataCommand *parent)
    : DataCommand(parent)
	, parentObject_(parentObject)
	, value_(value)
{
    // Check for validity //
    //
	OpenScenario::oscObjectBase::MemberMap members = parentObject_->getMembers();
	member_ = members[name];
    if ((name == "") || !member_)
    {
        setInvalid(); // Invalid
        setText(QObject::tr("AddOSCObjectCommand: Internal error! No name specified or member already present."));
        return;
    }
    else
    {
        setValid();
        setText(QObject::tr("AddOSCObject"));
    }

	type_ = member_->getType();
	if (type_ != OpenScenario::oscMemberValue::ENUM)
		{
        setInvalid(); // Invalid
        setText(QObject::tr("AddOSCObjectCommand: Internal error! Wrong type of value specified."));
        return;
    }


	openScenarioBase_ = parentObject_->getBase();
}

/*! \brief .
*
*/
AddOSCEnumValueCommand::~AddOSCEnumValueCommand()
{
    // Clean up //
    //
    if (isUndone())
    {
//        delete object_;
    }
    else
    {
        // nothing to be done (object is now owned by the road)
    }
}

/*! \brief .
*
*/
void
AddOSCEnumValueCommand::redo()
{
	OpenScenario::oscMemberValue *v = oscFactories::instance()->valueFactory->create(type_);

	if(v)
	{
		oscEnumValue *ev = dynamic_cast<oscEnumValue *>(v);
		if(ev)
		{
			ev->setValue(value_);
		}
	}
	member_->setValue(v);

	setRedone();
}

/*! \brief
*
*/
void
AddOSCEnumValueCommand::undo()
{
	const OpenScenario::oscObjectBase *obj = member_->getObject();
//	member_->setValue(NULL);
	delete obj;

   setUndone();
}


//#########################//
// RemoveOSCValueCommand //
//#########################//

/*RemoveOSCValueCommand::RemoveOSCValueCommand(OSCElement *element, DataCommand *parent) // or oscObjectBase ??
    : DataCommand(parent)
	, element_(element)
{
    // Check for validity //
    //
    if (!element_)
    {
        setInvalid(); // Invalid
        setText(QObject::tr("RemoveOSCValueCommand: Internal error! No element specified."));
        return;
    }
    else
    {
        setValid();
        setText(QObject::tr("RemoveOSCObject"));
    }
	oscBase_ = element_->getOSCBase();
	openScenarioBase_ = oscBase_->getOpenScenarioBase();
	object_ = element_->getObject();
}

/*! \brief .
*
*/
/*RemoveOSCValueCommand::~RemoveOSCValueCommand()
{
    // Clean up //
    //
    if (isUndone())
    {
        // nothing to be done (object is now owned by the road)
    }
    else
    {
//        delete object_;
    }
}

/*! \brief .
*
*/
/*void
RemoveOSCValueCommand::redo()
{
    element_->setObjectBase(NULL);				// todo: delete OpenScenario object/member
	oscBase_->delOSCElement(element_);

	element_->addOSCElementChanges(DataElement::CDE_DataElementDeleted);

    setRedone();
}

/*! \brief
*
*/
/*void
RemoveOSCValueCommand::undo()
{
    element_->setObjectBase(object_);
	oscBase_->addOSCElement(element_);

	element_->addOSCElementChanges(DataElement::CDE_DataElementAdded);

    setUndone();
}*/




