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
class oscCatalog;
template<typename T>
class oscValue;
class oscSourceFile;
}

class OSCBase;

#define BOOST_FILESYSTEM_NO_DEPRECATED
#include <boost/filesystem.hpp>


namespace bf = boost::filesystem;

//#########################//
// LoadOSCCatalogObjectCommand //
//#########################//

class LoadOSCCatalogObjectCommand : public DataCommand
{
public:
	explicit LoadOSCCatalogObjectCommand(OpenScenario::oscCatalog *catalog, int refId, OSCBase *base, OSCElement *element, DataCommand *parent = NULL);
    virtual ~LoadOSCCatalogObjectCommand();

    virtual int id() const
    {
        return 0x1014;
    }

    virtual void undo();
    virtual void redo();

private:
    LoadOSCCatalogObjectCommand(); /* not allowed */
    LoadOSCCatalogObjectCommand(const LoadOSCCatalogObjectCommand &); /* not allowed */
    LoadOSCCatalogObjectCommand &operator=(const LoadOSCCatalogObjectCommand &); /* not allowed */

private:
	OpenScenario::oscCatalog * catalog_;
	OpenScenario::oscObjectBase *objectBase_;
	int refId_;
	OSCElement *oscElement_;
	OSCBase *oscBase_;
};

//#########################//
// AddOSCCatalogObjectCommand //
//#########################//

class AddOSCCatalogObjectCommand : public DataCommand
{
public:
	explicit AddOSCCatalogObjectCommand(OpenScenario::oscCatalog *catalog, int refId, OpenScenario::oscObjectBase *objectBase, const std::string &path, OSCBase *base, OSCElement *element, DataCommand *parent = NULL);
    virtual ~AddOSCCatalogObjectCommand();

    virtual int id() const
    {
        return 0x1014;
    }

    virtual void undo();
    virtual void redo();

private:
    AddOSCCatalogObjectCommand(); /* not allowed */
    AddOSCCatalogObjectCommand(const AddOSCCatalogObjectCommand &); /* not allowed */
    AddOSCCatalogObjectCommand &operator=(const AddOSCCatalogObjectCommand &); /* not allowed */

private:
	OpenScenario::oscCatalog * catalog_;
    std::string path_;
	int refId_;
	OpenScenario::oscObjectBase *objectBase_;
	OSCElement *oscElement_;
	OSCBase *oscBase_;
};

//#########################//
// RemoveOSCCatalogObjectCommand //
//#########################//

class RemoveOSCCatalogObjectCommand : public DataCommand
{
public:
    explicit RemoveOSCCatalogObjectCommand(OpenScenario::oscCatalog *catalog, int refId, OSCElement *element, DataCommand *parent = NULL);
    virtual ~RemoveOSCCatalogObjectCommand();

    virtual int id() const
    {
        return 0x1015;
    }

    virtual void undo();
    virtual void redo();

private:
    RemoveOSCCatalogObjectCommand(); /* not allowed */
    RemoveOSCCatalogObjectCommand(const RemoveOSCCatalogObjectCommand &); /* not allowed */
    RemoveOSCCatalogObjectCommand &operator=(const RemoveOSCCatalogObjectCommand &); /* not allowed */

private:
	OpenScenario::oscObjectBase * oscObject_;
    OpenScenario::oscCatalog * catalog_;
	OSCElement *element_;
	OSCBase *oscBase_;
	bf::path path_;

	int refId_;
};

//#########################//
// AddOSCArrayMemberCommand //
//#########################//

class AddOSCArrayMemberCommand : public DataCommand
{
public:
	explicit AddOSCArrayMemberCommand(OpenScenario::oscArrayMember *arrayMember, OpenScenario::oscObjectBase *objectBase, OpenScenario::oscObjectBase *object, const std::string &name, OSCBase *base, OSCElement *element, DataCommand *parent = NULL);
    virtual ~AddOSCArrayMemberCommand();

    virtual int id() const
    {
        return 0x1014;
    }

    virtual void undo();
    virtual void redo();

private:
    AddOSCArrayMemberCommand(); /* not allowed */
    AddOSCArrayMemberCommand(const AddOSCArrayMemberCommand &); /* not allowed */
    AddOSCArrayMemberCommand &operator=(const AddOSCArrayMemberCommand &); /* not allowed */

private:
	OpenScenario::oscArrayMember *arrayMember_;
	OpenScenario::oscObjectBase *objectBase_, *object_;
	OpenScenario::oscMember *ownMember_;

    std::string typeName_;

	OSCBase *oscBase_;
	OSCElement *oscElement_;
};

//#########################//
// RemoveOSCArrayMemberCommand //
//#########################//

class RemoveOSCArrayMemberCommand : public DataCommand
{
public:
    explicit RemoveOSCArrayMemberCommand(OpenScenario::oscArrayMember *arrayMember, OpenScenario::oscObjectBase *objectBase, int index, OSCElement *element, DataCommand *parent = NULL);
    virtual ~RemoveOSCArrayMemberCommand();

    virtual int id() const
    {
        return 0x1015;
    }

    virtual void undo();
    virtual void redo();

private:
    RemoveOSCArrayMemberCommand(); /* not allowed */
    RemoveOSCArrayMemberCommand(const RemoveOSCArrayMemberCommand &); /* not allowed */
    RemoveOSCArrayMemberCommand &operator=(const RemoveOSCArrayMemberCommand &); /* not allowed */

private:
	OpenScenario::oscArrayMember *arrayMember_;
	OpenScenario::oscObjectBase *objectBase_, *oscObject_;

    int index_;

	OSCBase *oscBase_;
	OSCElement *oscElement_;
};

//#########################//
// AddOSCObjectCommand //
//#########################//

class AddOSCObjectCommand : public DataCommand
{
public:
	explicit AddOSCObjectCommand(OpenScenario::oscObjectBase *parentObject, OSCBase *oscBase, const std::string &name, OSCElement *element, OpenScenario::oscSourceFile *file, DataCommand *parent = NULL);
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
	OpenScenario::oscObjectBase * parentObject_;

	OSCElement *element_;
	OSCBase *oscBase_;
	OpenScenario::oscSourceFile *sourceFile_;
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
        return 0x1012;
    }

    virtual void undo();
    virtual void redo();

private:
    RemoveOSCObjectCommand(); /* not allowed */
    RemoveOSCObjectCommand(const RemoveOSCObjectCommand &); /* not allowed */
    RemoveOSCObjectCommand &operator=(const RemoveOSCObjectCommand &); /* not allowed */

private:
	const OpenScenario::OpenScenarioBase * openScenarioBase_;
    OpenScenario::oscObjectBase *object_;
	OpenScenario::oscMember *parentMember_;

	OSCBase *oscBase_;
	OSCElement *element_;
};

//#########################//
// ChangeOSCObjectChoiceCommand //
//#########################//

class ChangeOSCObjectChoiceCommand : public DataCommand
{
public:
	explicit ChangeOSCObjectChoiceCommand(OpenScenario::oscObjectBase *parentObject, OpenScenario::oscMember *oldChosenMember, OpenScenario::oscMember *newChosenMember, OSCElement *element, DataCommand *parent = NULL);
    virtual ~ChangeOSCObjectChoiceCommand();

    virtual int id() const
    {
        return 0x1011;
    }

    virtual void undo();
    virtual void redo();

private:
    ChangeOSCObjectChoiceCommand(); /* not allowed */
    ChangeOSCObjectChoiceCommand(const ChangeOSCObjectChoiceCommand &); /* not allowed */
    ChangeOSCObjectChoiceCommand &operator=(const ChangeOSCObjectChoiceCommand &); /* not allowed */

private:
	OpenScenario::oscObjectBase *parentObject_;
	OpenScenario::oscMember *oldChosenMember_;
	OpenScenario::oscMember *newChosenMember_;

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
        return 0x1013;
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

#if 0
class RemoveOSCValueCommand : public DataCommand
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
    RemoveOSCValueCommand(const RemoveOSCValueCommand &); /* not allowed */
    RemoveOSCValueCommand &operator=(const RemoveOSCValueCommand &); /* not allowed */

private:
	const OpenScenario::OpenScenarioBase * openScenarioBase_;
    const OpenScenario::oscObjectBase *object_;

	OSCBase *oscBase_;
	OSCElement *element_;
};
#endif

//#########################//
// SetOSCObjectPropertiesCommand //
//#########################//
template<typename T>
class SetOSCValuePropertiesCommand : public DataCommand
{
public:
	explicit SetOSCValuePropertiesCommand(OSCElement *element, OpenScenario::oscObjectBase *object, const std::string &memberName, const T &value, DataCommand *parent = NULL)
	{
		memberName_ = memberName;
	// Check for validity //
    //
    if (!element || !object)
    {
        setInvalid(); // Invalid
        setText(QObject::tr("SetOSCValuePropertiesCommand: Internal error! No OSCElement or no OpenScenario object specified."));
        return;
    }

	element_ = element;

	newOSCValue_ = value;
	OpenScenario::oscMember *member = object->getMember(memberName);
	v_ = member->getValue();
	
	if (!v_)
	{
		v_ = member->createValue();

		if(member->getType() == oscMemberValue::ENUM)
		{
			oscEnumValue *ev = dynamic_cast<oscEnumValue *>(v_);
			oscEnum *em = dynamic_cast<oscEnum *>(member);
			if(ev && em)
			{
				ev->enumType = em->enumType;
			}
		}
	}

	OpenScenario::oscValue<T> *oscTypeMemberValue = dynamic_cast<OpenScenario::oscValue<T> *>(v_);
	if (oscTypeMemberValue)
	{
		oldOSCValue_ = oscTypeMemberValue->getValue();
	}

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
		element_->addOSCElementChanges(OSCElement::COE_ParameterChange);

		setUndone();
	}

    virtual void redo()
	{
		v_->setValue(newOSCValue_);
		element_->addOSCElementChanges(OSCElement::COE_ParameterChange);

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
