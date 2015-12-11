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

using namespace OpenScenario;

//#########################//
// AddOSCObjectCommand //
//#########################//

AddOSCObjectCommand::AddOSCObjectCommand(OpenScenario::oscObjectBase *oscBase, const std::string &name, OSCElement *element, DataCommand *parent)
    : DataCommand(parent)
    , memberName_(name)
	, oscBase_(oscBase)
	, element_(element)
	, object_(NULL)
{
    // Check for validity //
    //
    if (name == "")
    {
        setInvalid(); // Invalid
        setText(QObject::tr("AddOSCObjectCommand: Internal error! No name specified."));
        return;
    }
    else
    {
        setValid();
        setText(QObject::tr("AddOSCObject"));
    }

	base_ = element_->getOSCBase();
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

/*	oscMember *m = members[memberName_];
	std::string memTypeName = m->getTypeName();
	object_ = oscBase->getObjectFactory()->create(memTypeName);
	object_->initialize(oscBase);
	m->setValue(object_);
	*/
	element_->setObject(object_);
	base_->addOSCElement(element_);

	element_->addOSCElementChanges(DataElement::CDE_DataElementAdded);

    setRedone();
}

/*! \brief
*
*/
void
AddOSCObjectCommand::undo()
{
//  oscBase->removeObject(object_);
	element_->setObject(NULL);
	base_->delOSCElement(element_);

	element_->addOSCElementChanges(DataElement::CDE_DataElementDeleted);

   setUndone();
}

//#########################//
// RemoveOSCObjectCommand //
//#########################//

RemoveOSCObjectCommand::RemoveOSCObjectCommand(OpenScenario::oscObjectBase * oscBase, OpenScenario::oscObject *object, OSCElement *element, DataCommand *parent) // or oscObjectBase ??
    : DataCommand(parent)
    , object_(object)
	, element_(element)
    , oscBase_(oscBase)
{
    // Check for validity //
    //
    if (!object)
    {
        setInvalid(); // Invalid
        setText(QObject::tr("RemoveOSCObjectCommand: Internal error! No object specified."));
        return;
    }
    else
    {
        setValid();
        setText(QObject::tr("RemoveOSCObject"));
    }
	base_ = element->getOSCBase();
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
    //oscBase_->removeObject(object_); 
	base_->delOSCElement(element_);

	element_->addOSCElementChanges(DataElement::CDE_DataElementDeleted);

    setRedone();
}

/*! \brief
*
*/
void
RemoveOSCObjectCommand::undo()
{
   // oscBase_->addObject(object_);
	base_->addOSCElement(element_);

	element_->addOSCElementChanges(DataElement::CDE_DataElementAdded);

    setUndone();
}



