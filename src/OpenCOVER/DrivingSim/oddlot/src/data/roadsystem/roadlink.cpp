/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   11/2/2010
**
**************************************************************************/

#include "roadlink.hpp"

#include "rsystemelementroad.hpp"
#include "roadsystem.hpp"

RoadLink::RoadLink(const QString &elementType, const QString &elementId, const QString &contactPoint)
    : DataElement()
    , roadLinkChange_(0x0)
    , parentRoad_(NULL)
    , type_(DRL_UNKNOWN)
    , elementType_(elementType)
    , elementId_(elementId)
    , contactPoint_(contactPoint)
{
}

RoadLink::~RoadLink()
{
}

//################//
// RoadSystem     //
//################//

void
RoadLink::setParentRoad(RSystemElementRoad *road, RoadLinkType type)
{
    setParentElement(road);
    type_ = type;
    parentRoad_ = road;
    addRoadLinkChanges(RoadLink::CRL_ParentRoadChanged);
}

bool
RoadLink::isLinkValid() const
{
    if (parentRoad_)
    {
        if (elementType_ == "junction")
        {
            if (parentRoad_->getRoadSystem()->getJunction(elementId_))
            {
                return true;
            }
        }
        else if (elementType_ == "road")
        {
            if (parentRoad_->getRoadSystem()->getRoad(elementId_))
            {
                return true;
            }
        }
        else if (elementType_ == "fiddleyard")
        {
            if (parentRoad_->getRoadSystem()->getFiddleyard(elementId_))
            {
                return true;
            }
        }
    }

    return false;
}

//################//
// RoadLink //
//################//

void
RoadLink::setElementId(const QString &elementId)
{
    elementId_ = elementId;
    addRoadLinkChanges(RoadLink::CRL_IdChanged);
    if (parentRoad_)
    {
        if (type_ == DRL_PREDECESSOR)
        {
            parentRoad_->addRoadChanges(RSystemElementRoad::CRD_PredecessorChange);
        }
        else if (type_ == DRL_SUCCESSOR)
        {
            parentRoad_->addRoadChanges(RSystemElementRoad::CRD_SuccessorChange);
        }
    }
}

void
RoadLink::setElementType(const QString &elementType)
{
    elementType_ = elementType;
    addRoadLinkChanges(RoadLink::CRL_TypeChanged);
    if (parentRoad_)
    {
        if (type_ == DRL_PREDECESSOR)
        {
            parentRoad_->addRoadChanges(RSystemElementRoad::CRD_PredecessorChange);
        }
        else if (type_ == DRL_SUCCESSOR)
        {
            parentRoad_->addRoadChanges(RSystemElementRoad::CRD_SuccessorChange);
        }
    }
}

void
RoadLink::setContactPoint(const QString &contactPoint)
{
    contactPoint_ = contactPoint;
    addRoadLinkChanges(RoadLink::CRL_ContactPointChanged);
    if (parentRoad_)
    {
        if (type_ == DRL_PREDECESSOR)
        {
            parentRoad_->addRoadChanges(RSystemElementRoad::CRD_PredecessorChange);
        }
        else if (type_ == DRL_SUCCESSOR)
        {
            parentRoad_->addRoadChanges(RSystemElementRoad::CRD_SuccessorChange);
        }
    }
}

//##################//
// Observer Pattern //
//##################//

/*! \brief Called after all Observers have been notified.
*
* Resets the change flags to 0x0.
*/
void
RoadLink::notificationDone()
{
    roadLinkChange_ = 0x0;
    DataElement::notificationDone(); // pass to base class
}

/*! \brief Add one or more change flags.
*
*/
void
RoadLink::addRoadLinkChanges(int changes)
{
    if (changes)
    {
        roadLinkChange_ |= changes;
        notifyObservers();
    }
}

//###################//
// Visitor Pattern   //
//###################//

/*!
* Accepts a visitor.
*/
void
RoadLink::accept(Visitor *visitor)
{
    visitor->visit(this);
}
