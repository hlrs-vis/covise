/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   02.02.2010
**
**************************************************************************/

#include "rsystemelementjunctiongroup.hpp"


RSystemElementJunctionGroup::RSystemElementJunctionGroup(const QString &name, const odrID &id, const QString &type)
    : RSystemElement(name, id, RSystemElement::DRE_JunctionGroup)
	, type_(type)
    , junctionGroupChanges_(0x0)
{
}

RSystemElementJunctionGroup::~RSystemElementJunctionGroup()
{
}


void 
RSystemElementJunctionGroup::addJunction(const QString junctionReference)
{
	if (!junctionReferences_.contains(junctionReference))
	{
		junctionReferences_.append(junctionReference);
	}
}

bool 
RSystemElementJunctionGroup::delJunction(const QString junctionReference)
{
	return junctionReferences_.removeOne(junctionReference);
}

void
RSystemElementJunctionGroup::clearReferences()
{
	junctionReferences_.clear();
}


//##################//
// Observer Pattern //
//##################//

/*! \brief Called after all Observers have been notified.
*
* Resets the change flags to 0x0.
*/
void
RSystemElementJunctionGroup::notificationDone()
{
    junctionGroupChanges_ = 0x0;
    RSystemElement::notificationDone();
}

/*! \brief Add one or more change flags.
*
*/
void
RSystemElementJunctionGroup::addJunctionGroupChanges(int changes)
{
    if (changes)
    {
        junctionGroupChanges_ |= changes;
        notifyObservers();
    }
}

//###################//
// Prototype Pattern //
//###################//

/*! \brief Creates and returns a deep copy clone of this object.
*
*/
RSystemElementJunctionGroup *
RSystemElementJunctionGroup::getClone()
{
    // New Junction //
    //
    RSystemElementJunctionGroup *clonedJunctionGroup = new RSystemElementJunctionGroup(getName(), getID(), type_);

    return clonedJunctionGroup;
}

//###################//
// Visitor Pattern   //
//###################//

/*!
* Accepts a visitor and passes it to all child
* nodes if autoTraverse is true.
*/
void
RSystemElementJunctionGroup::accept(Visitor *visitor)
{
    visitor->visit(this);
}

