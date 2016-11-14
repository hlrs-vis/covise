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
#include "oscCatalog.h"

using namespace OpenScenario;

//#########################//
// LoadOSCCatalogObjectCommand //
//#########################//

LoadOSCCatalogObjectCommand::LoadOSCCatalogObjectCommand(OpenScenario::oscCatalog *catalog, int refId, OSCBase *base, OSCElement *element, DataCommand *parent)
    : DataCommand(parent)
	, catalog_(catalog)
	, refId_(refId)
	, oscElement_(element)
	, oscBase_(base)
{
    // Check for validity //
    //
	objectBase_ = catalog_->getCatalogObject(refId_);
	if (!catalog_ || objectBase_)
    {
        setInvalid(); // Invalid
        setText(QObject::tr("LoadOSCCatalogObjectCommand: Internal error! Member or Id not valid."));
        return;
    }
    else
    {
        setValid();
        setText(QObject::tr("AddOSCCatalogObject"));
    }

}

/*! \brief .
*
*/
LoadOSCCatalogObjectCommand::~LoadOSCCatalogObjectCommand()
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
LoadOSCCatalogObjectCommand::redo()
{
	catalog_->fullReadCatalogObjectWithName(refId_);

	oscElement_->setObjectBase(catalog_->getCatalogObject(refId_));
	oscBase_->addOSCElement(oscElement_);
	
	setRedone();
}

/*! \brief
*
*/
void
	LoadOSCCatalogObjectCommand::undo()
{
	catalog_->removeCatalogObject(refId_);

	oscElement_->setObjectBase(NULL);
	oscBase_->delOSCElement(oscElement_);

	setUndone();
}


//#########################//
// AddOSCCatalogObjectCommand //
//#########################//

AddOSCCatalogObjectCommand::AddOSCCatalogObjectCommand(OpenScenario::oscCatalog *catalog, int refId, OpenScenario::oscObjectBase *objectBase, const std::string &path, OSCBase *base, OSCElement *element, DataCommand *parent)
    : DataCommand(parent)
	, catalog_(catalog)
	, refId_(refId)
	, objectBase_(objectBase)
	, path_(path)
	, oscElement_(element)
	, oscBase_(base)
{
    // Check for validity //
    //

	if (!catalog_ || (objectBase_ && !oscBase_))
    {
        setInvalid(); // Invalid
        setText(QObject::tr("AddOSCCatalogObjectCommand: Internal error! Member not valid."));
        return;
    }
    else
    {
        setValid();
        setText(QObject::tr("AddOSCCatalogObject"));
    }


}

/*! \brief .
*
*/
AddOSCCatalogObjectCommand::~AddOSCCatalogObjectCommand()
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
AddOSCCatalogObjectCommand::redo()
{

	if (objectBase_)
	{
		// create refId //
		//
		OpenScenario::oscMember *member = objectBase_->getMember("refId");
		OpenScenario::oscMemberValue *v = member->getOrCreateValue();
		v->setValue(refId_); 
		catalog_->addCatalogObject(refId_, objectBase_, boost::filesystem::path(path_));

		oscElement_->setObjectBase(objectBase_);
		oscBase_->addOSCElement(oscElement_);
	}
	else
	{
		catalog_->addObjToObjectsMap(refId_, path_, NULL);
	}
	
	setRedone();
}

/*! \brief
*
*/
void
	AddOSCCatalogObjectCommand::undo()
{
/*	OpenScenario::oscMember *member = objectBase_->getMember("refId");
	member->deleteValue(); */

	if (objectBase_)
	{
		catalog_->removeCatalogObject(refId_);

		oscElement_->setObjectBase(NULL);
		oscBase_->delOSCElement(oscElement_);
	}
	else
	{
		catalog_->removeObjFromObjectsMap(refId_);
	}

	setUndone();
}

//#########################//
// RemoveOSCCatalogObjectCommand //
//#########################//

RemoveOSCCatalogObjectCommand::RemoveOSCCatalogObjectCommand(OpenScenario::oscCatalog *catalog, int refId, OSCElement *element ,DataCommand *parent) // or oscObjectBase ??
    : DataCommand(parent)
	, catalog_(catalog)
	, refId_(refId)
	, element_(element)
	, oscBase_(NULL)
{
    // Check for validity //
    //
	
	if (!catalog_ )
    {
        setInvalid(); // Invalid
        setText(QObject::tr("RemoveOSCCatalogObjectCommand: Internal error! No valid member or object ID specified."));
        return;
    }
    else
    {
        setValid();
        setText(QObject::tr("RemoveOSCCatalogObject"));
    }

	if (element_)
	{
		oscBase_ = element_->getOSCBase();
	}

	oscObject_ = catalog_->getCatalogObject(refId_);
	path_ = catalog_->getPath(refId_);
}

/*! \brief .
*
*/
RemoveOSCCatalogObjectCommand::~RemoveOSCCatalogObjectCommand()
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
RemoveOSCCatalogObjectCommand::redo()
{
	if (oscBase_)
	{
		oscBase_->delOSCElement(element_);
	}

	catalog_->removeCatalogObject(refId_);

    setRedone();
}

/*! \brief
*
*/
void
RemoveOSCCatalogObjectCommand::undo()
{
	catalog_->addCatalogObject(refId_, oscObject_, path_);
	element_->setObjectBase(oscObject_);
	oscBase_->addOSCElement(element_);

    setUndone();
}

//#########################//
// AddOSCArrayMemberCommand //
//#########################//

AddOSCArrayMemberCommand::AddOSCArrayMemberCommand(OpenScenario::oscArrayMember *arrayMember, OpenScenario::oscObjectBase *objectBase,  OpenScenario::oscObjectBase *object, const std::string &name, OSCBase *base, OSCElement *element, DataCommand *parent)
    : DataCommand(parent)
	, arrayMember_(arrayMember)
	, typeName_(name)
	, objectBase_(objectBase)
	, oscElement_(element)
	, oscBase_(base)
	, object_(object)
	, ownMember_(NULL)
{
    // Check for validity //
    //

	if (!arrayMember_ || !objectBase_)
    {
        setInvalid(); // Invalid
        setText(QObject::tr("AddOSCArrayMemberCommand: Internal error! Member not valid."));
        return;
    }
    else
    {
        setValid();
        setText(QObject::tr("AddOSCArrayMember"));
    }

	if (object_)
	{
		ownMember_ = arrayMember_->getObject()->getMember(typeName_);
	}
}

/*! \brief .
*
*/
AddOSCArrayMemberCommand::~AddOSCArrayMemberCommand()
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
AddOSCArrayMemberCommand::redo()
{
	if (!object_)
	{
		object_ = objectBase_->getMember(typeName_)->createObject();
		ownMember_ = arrayMember_->getObject()->getMember(typeName_);
	}

	if(object_)
	{
		oscElement_->setObjectBase(object_);
		oscBase_->addOSCElement(oscElement_);

		object_->setOwnMember(ownMember_);
		arrayMember_->push_back(object_);
	}
	
	setRedone();
}

/*! \brief
*
*/
void
	AddOSCArrayMemberCommand::undo()
{
	arrayMember_->erase(arrayMember_->end() - 1);

	oscBase_->delOSCElement(oscElement_);
	oscElement_->setObjectBase(NULL);

	setUndone();
}

//#########################//
// RemoveOSCArrayMemberCommand //
//#########################//

RemoveOSCArrayMemberCommand::RemoveOSCArrayMemberCommand(OpenScenario::oscArrayMember *arrayMember, OpenScenario::oscObjectBase *objectBase, int index, OSCElement *element, DataCommand *parent) 
    : DataCommand(parent)
	, arrayMember_(arrayMember)
	, index_(index)
	, objectBase_(objectBase)
	, oscElement_(element)
	, oscBase_(NULL)
{
    // Check for validity //
    //
	oscObject_ = arrayMember_->at(index_);

	if (!arrayMember_ || !objectBase_ || !oscObject_)
    {
        setInvalid(); // Invalid
        setText(QObject::tr("RemoveOSCArrayMemberCommand: Internal error! No valid member or object ID specified."));
        return;
    }
    else
    {
        setValid();
        setText(QObject::tr("RemoveOSCArrayMember"));
    }

	if (oscElement_)
	{
		oscBase_ = oscElement_->getOSCBase();
	}
}

/*! \brief .
*
*/
RemoveOSCArrayMemberCommand::~RemoveOSCArrayMemberCommand()
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
RemoveOSCArrayMemberCommand::redo()
{
	if (oscBase_)
	{
		oscBase_->delOSCElement(oscElement_);
	}

	arrayMember_->erase(arrayMember_->begin() + index_);

    setRedone();
}

/*! \brief
*
*/
void
RemoveOSCArrayMemberCommand::undo()
{
	oscElement_->setObjectBase(oscObject_);
	oscBase_->addOSCElement(oscElement_);

	arrayMember_->emplace(arrayMember_->begin() + index_, oscObject_);

	setUndone();
}

//#########################//
// AddOSCObjectCommand //
//#########################//

AddOSCObjectCommand::AddOSCObjectCommand(OpenScenario::oscObjectBase *parentObject, OSCBase *base, const std::string &name, OSCElement *element, OpenScenario::oscSourceFile *file, DataCommand *parent)
    : DataCommand(parent)
	, element_(element)
	, parentObject_(parentObject)
	, oscBase_(base)
	, sourceFile_(file)
{
    // Check for validity //
    //
	OpenScenario::oscObjectBase::MemberMap members = parentObject_->getMembers();
	member_ = members[name];
    if (name == "")
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

	if (member_)	// there is no member for a catalog
	{
		typeName_ = member_->getTypeName();
	}
	else
	{
		typeName_ = name;
	}

	openScenarioBase_ = base->getOpenScenarioBase();
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
		obj->initialize(openScenarioBase_, parentObject_, member_, sourceFile_);	
		if (member_)
		{
			member_->setValue(obj);
		}

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
	if (member_)
	{
		member_->deleteValue();
	}

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
// ChangeOSCObjectChoiceCommand //
//#########################//

ChangeOSCObjectChoiceCommand::ChangeOSCObjectChoiceCommand(OpenScenario::oscObjectBase *parentObject, OpenScenario::oscMember *oldChosenMember, OpenScenario::oscMember *newChosenMember, OSCElement *element, DataCommand *parent)
    : DataCommand(parent)
	, parentObject_(parentObject)
	, oldChosenMember_(oldChosenMember)
	, newChosenMember_(newChosenMember)
	, element_(element)
{
    // Check for validity //
    //
	if (!parentObject_ || !oldChosenMember_ || !newChosenMember_ || !element_)
    {
        setInvalid(); // Invalid
        setText(QObject::tr("ChangeOSCObjectChoiceCommand: Internal error! No valid objects."));
        return;
    }
    else
    {
        setValid();
        setText(QObject::tr("ChangeOSCObjectChoiceCommand"));
    }
}

/*! \brief .
*
*/
ChangeOSCObjectChoiceCommand::~ChangeOSCObjectChoiceCommand()
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
ChangeOSCObjectChoiceCommand::redo()
{
	parentObject_->setChosenMember(newChosenMember_);
	element_->addOSCElementChanges(OSCElement::COE_ChoiceChanged);
	
	setRedone();
}

/*! \brief
*
*/
void
ChangeOSCObjectChoiceCommand::undo()
{
	parentObject_->setChosenMember(oldChosenMember_);
	element_->addOSCElementChanges(OSCElement::COE_ChoiceChanged);

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
	member_ = parentObject_->getMember(name);
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


#if 0
//#########################//
// RemoveOSCValueCommand //
//#########################//

RemoveOSCValueCommand::RemoveOSCValueCommand(OSCElement *element, DataCommand *parent) // or oscObjectBase ??
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
RemoveOSCValueCommand::~RemoveOSCValueCommand()
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
void
RemoveOSCValueCommand::undo()
{
    element_->setObjectBase(object_);
	oscBase_->addOSCElement(element_);

	element_->addOSCElementChanges(DataElement::CDE_DataElementAdded);

    setUndone();
}
#endif




