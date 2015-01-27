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
RSystemElement::RSystemElement(const QString &name, const QString &id, DRoadSystemElementType elementType)
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
    QString text = id_;
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
    addRSystemElementChanges(RSystemElement::CRE_NameChange);
}

void
RSystemElement::setID(const QString &id)
{
    id_ = id;
    addRSystemElementChanges(RSystemElement::CRE_IdChange);
}

QString
RSystemElement::getNewId(RSystemElement *element, QString &name)
{
    QStringList parts = element->getID().split("_");
    QString newId = parts.at(0) + "_" + parts.at(1) + "_" + name;

    return newId;
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
