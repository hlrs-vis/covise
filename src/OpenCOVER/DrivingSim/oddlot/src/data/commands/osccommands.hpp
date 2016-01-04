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

#ifndef OSCOBJECTCOMMANDS_HPP
#define OSCOBJECTCOMMANDS_HPP


#include "datacommand.hpp"

// Data //
//
#include "src/data/oscsystem/oscelement.hpp"

#include <QMap>

namespace OpenScenario
{
class oscObjectBase;
class OpenScenarioBase;
class oscMember;
class oscMemberValue;
template<typename T>
class oscValue;
}

class OSCBase;


//#########################//
// AddOSCObjectCommand //
//#########################//

class AddOSCObjectCommand : public DataCommand
{
public:
	explicit AddOSCObjectCommand(const OpenScenario::oscObjectBase *parentObject, OSCBase *oscBase, const std::string &name, OSCElement *element, DataCommand *parent = NULL);
    virtual ~AddOSCObjectCommand();

    virtual int id() const
    {
        return 0x1011;
    }

    virtual void undo();
    virtual void redo();

private:
    AddOSCObjectCommand(); /* not allowed */
    AddOSCObjectCommand(const AddOSCObjectCommand &); /* not allowed */
    AddOSCObjectCommand &operator=(const AddOSCObjectCommand &); /* not allowed */

private:
	OpenScenario::OpenScenarioBase * openScenarioBase_;
    std::string typeName_;
	const OpenScenario::oscObjectBase * parentObject_;

	OSCElement *element_;
	OSCBase *oscBase_;
	OpenScenario::oscMember *member_;
};

//#########################//
// RemoveOSCObjectCommand //
//#########################//

class RemoveOSCObjectCommand : public DataCommand
{
public:
    explicit RemoveOSCObjectCommand(OSCElement *element,DataCommand *parent = NULL);
    virtual ~RemoveOSCObjectCommand();

    virtual int id() const
    {
        return 0x1011;
    }

    virtual void undo();
    virtual void redo();

private:
    RemoveOSCObjectCommand(); /* not allowed */
    RemoveOSCObjectCommand(const RemoveOSCObjectCommand &); /* not allowed */
    RemoveOSCObjectCommand &operator=(const RemoveOSCObjectCommand &); /* not allowed */

private:
	const OpenScenario::OpenScenarioBase * openScenarioBase_;
    const OpenScenario::oscObjectBase *object_;

	OSCBase *oscBase_;
	OSCElement *element_;
};

//#########################//
// AddOSCEnumValueCommand //
//#########################//

class AddOSCEnumValueCommand : public DataCommand
{
public:
	explicit AddOSCEnumValueCommand(const OpenScenario::oscObjectBase *parentObject, const std::string &name, int value, DataCommand *parent = NULL);
    virtual ~AddOSCEnumValueCommand();

    virtual int id() const
    {
        return 0x1011;
    }

    virtual void undo();
    virtual void redo();

private:
    AddOSCEnumValueCommand(); /* not allowed */
    AddOSCEnumValueCommand(const AddOSCEnumValueCommand &); /* not allowed */
    AddOSCEnumValueCommand &operator=(const AddOSCEnumValueCommand &); /* not allowed */

private:
	OpenScenario::OpenScenarioBase * openScenarioBase_;
	OpenScenario::oscMemberValue::MemberTypes type_;
	const OpenScenario::oscObjectBase * parentObject_;

	OSCElement *element_;
	OpenScenario::oscMember *member_;
	int value_;
};

//#########################//
// AddOSCValueCommand //
//#########################//
template<typename T>
class AddOSCValueCommand : public DataCommand
{
public:
	explicit AddOSCValueCommand(const OpenScenario::oscObjectBase *parentObject, const std::string &name, T &value, DataCommand *parent = NULL)
		:DataCommand(parent)
{

	parentObject_ = parentObject;
	value_ = value;
    // Check for validity //
    //
	OpenScenario::oscObjectBase::MemberMap members = parentObject_->getMembers();
	member_ = members[name];
    if ((name == "") || !member_)
    {
        setInvalid(); // Invalid
        setText(QObject::tr("AddOSCValueCommand: Internal error! No name specified or member already present."));
        return;
    }
    else
    {
        setValid();
        setText(QObject::tr("AddOSCObject"));
    }

	typeName_ = member_->getType();
}

    virtual ~AddOSCValueCommand()
		{
    if (isUndone())
    {
//        delete object_;
    }
    else
    {

    }
}

    virtual int id() const
    {
        return 0x1011;
    }

    virtual void undo()
	{
//	member_->setValue(NULL);

   setUndone();
}

    virtual void redo()
	{
	OpenScenario::oscMemberValue *v = oscFactories::instance()->valueFactory->create(typeName_);

	if(v)
	{
		v->setValue(value_);
		member_->setValue(v);
	}

	setRedone();
}

private:
    AddOSCValueCommand(); /* not allowed */
    AddOSCValueCommand(const AddOSCValueCommand &); /* not allowed */
    AddOSCValueCommand &operator=(const AddOSCValueCommand &); /* not allowed */

private:
	OpenScenario::OpenScenarioBase * openScenarioBase_;
	OpenScenario::oscMemberValue::MemberTypes typeName_;
	const OpenScenario::oscObjectBase * parentObject_;

	OpenScenario::oscMember *member_;
	T value_;
};

//#########################//
// RemoveOSCValueCommand //
//#########################//

/*class RemoveOSCValueCommand : public DataCommand
{
public:
    explicit RemoveOSCValueCommand(OSCElement *element,DataCommand *parent = NULL);
    virtual ~RemoveOSCValueCommand();

    virtual int id() const
    {
        return 0x1011;
    }

    virtual void undo();
    virtual void redo();

private:
    RemoveOSCValueCommand(); /* not allowed */
 /*   RemoveOSCValueCommand(const RemoveOSCValueCommand &); /* not allowed */
/*    RemoveOSCValueCommand &operator=(const RemoveOSCValueCommand &); /* not allowed */

/*private:
	const OpenScenario::OpenScenarioBase * openScenarioBase_;
    const OpenScenario::oscObjectBase *object_;

	OSCBase *oscBase_;
	OSCElement *element_;
};*/

//#########################//
// SetOSCObjectPropertiesCommand //
//#########################//
template<typename T>
class SetOSCValuePropertiesCommand : public DataCommand
{
public:
	explicit SetOSCValuePropertiesCommand(OSCElement *element, const std::string &memberName, const T &value, DataCommand *parent = NULL)
	{
		memberName_ = memberName;
	// Check for validity //
    //
    if (!element)
    {
        setInvalid(); // Invalid
        setText(QObject::tr("SetOSCValuePropertiesCommand: Internal error! No OSCElement specified."));
        return;
    }

	element_ = element;
	const OpenScenario::oscObjectBase *object = element_->getObject();
	newOSCValue_ = value;
	OpenScenario::oscMember *member = object->getMembers().at(memberName);
	v_ = member->getValue();
	if (!v_)
	{
		setInvalid(); // Invalid
		setText(QObject::tr("SetOSCValuePropertiesCommand: Internal error! No OSCElement specified."));
		return;
	}
//	oldOSCValue_ = v_->getValue();

	setValid();
	setText(QObject::tr("SetProperties"));

	}
    virtual ~SetOSCValuePropertiesCommand()
	{
		// Clean up //
    //
    if (isUndone())
    {
    }
    else
    {
    }
	}

    virtual int id() const
    {
        return 0x1011;
    }

    virtual void undo()
	{

	v_->setValue(oldOSCValue_);
	if (memberName_ == "name")
		{
			element_->addOSCElementChanges(OSCElement::COE_ParameterChange);
		}

    setUndone();
	}

    virtual void redo()
	{
		v_->setValue(newOSCValue_);
		if (memberName_ == "name")
		{
			element_->addOSCElementChanges(OSCElement::COE_ParameterChange);
		}
    setRedone();
	}

private:
    SetOSCValuePropertiesCommand(); /* not allowed */
    SetOSCValuePropertiesCommand(const SetOSCValuePropertiesCommand &); /* not allowed */
    SetOSCValuePropertiesCommand &operator=(const SetOSCValuePropertiesCommand &); /* not allowed */

private:
	OSCElement *element_;
	OpenScenario::oscMemberValue *v_;
	std::string memberName_;
	T newOSCValue_;
	T oldOSCValue_;
};

#endif // OSCOBJECTCOMMANDS_HPP
