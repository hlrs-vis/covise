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

#include "rsystemelement.hpp"

#include "roadsystem.hpp"

#include <QStringList>

/*! \brief The constructor does nothing special.
*
*/
RSystemElement::RSystemElement(const QString &name, const odrID &id, DRoadSystemElementType elementType)
    : DataElement()
    , elementType_(elementType)
    , parentRoadSystem_(NULL)
    , name_(name)
    , id_(id)
    , rSystemElementChanges_(0x0)
{
}

RSystemElement::~RSystemElement()
{
}

//################//
// RoadSystem     //
//################//

void
RSystemElement::setRoadSystem(RoadSystem *parentRoadSystem)
{
    setParentElement(parentRoadSystem);
    parentRoadSystem_ = parentRoadSystem;
    addRSystemElementChanges(RSystemElement::CRE_ParentRoadSystemChange);
}

//################//
// RSystemElement //
//################//

QString
RSystemElement::getIdName() const
{
    QString text = id_.speakingName();
    if (!name_.isEmpty())
    {
        text.append(" (");
        text.append(name_);
        text.append(")");
    }
    return text;
}

void
RSystemElement::setName(const QString &name)
{
    name_ = name;
	id_.setName(name);
    addRSystemElementChanges(RSystemElement::CRE_NameChange);
}

void
RSystemElement::setID(const odrID &id)
{
    id_ = id;
    addRSystemElementChanges(RSystemElement::CRE_IdChange);
}


//##################//
// Observer Pattern //
//##################//

/*! \brief Called after all Observers have been notified.
*
* Resets the change flags to 0x0.
*/
void
RSystemElement::notificationDone()
{
    rSystemElementChanges_ = 0x0;
    DataElement::notificationDone(); // pass to base class
}

/*! \brief Add one or more change flags.
*
*/
void
RSystemElement::addRSystemElementChanges(int changes)
{
    if (changes)
    {
        rSystemElementChanges_ |= changes;
        notifyObservers();
    }
}
