/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   22.03.2010
**
**************************************************************************/

#include "parkingspaceobject.hpp"

//###################//
// Static Functions  //
//###################//

ParkingSpaceMarking::ParkingSpaceMarkingSide
ParkingSpaceMarking::parseParkingSpaceMarkingSide(const QString &side)
{
	if (side == "front")
		return ParkingSpaceMarking::PSM_FRONT;
	else if (side == "rear")
		return ParkingSpaceMarking::PSM_REAR;
	else if (side == "left")
		return ParkingSpaceMarking::PSM_LEFT;
	else if (side == "right")
		return ParkingSpaceMarking::PSM_RIGHT;	
	else
	{
		qDebug("WARNING: unknown side type: %s", side.toUtf8().constData());
		return ParkingSpaceMarking::PSM_NONE;
	}
}

QString 
ParkingSpaceMarking::parseParkingSpaceMarkingSideBack(ParkingSpaceMarking::ParkingSpaceMarkingSide side)
{
	if (side == ParkingSpaceMarking::PSM_NONE)
		return QString("none");
	else if (side == ParkingSpaceMarking::PSM_FRONT)
		return QString("front");
	else if (side == ParkingSpaceMarking::PSM_REAR)
		return QString("rear");
	else if (side == ParkingSpaceMarking::PSM_LEFT)
		return QString("left");
	else if (side == ParkingSpaceMarking::PSM_RIGHT)
		return QString("right");
	else
	{
		qDebug("WARNING: unknown parking space marking type.");
		return QString("none");
	}
}

//####################//
// Constructors       //
//####################//

ParkingSpaceMarking::ParkingSpaceMarking(ParkingSpaceMarkingSide side, LaneRoadMark::RoadMarkType type, double width, LaneRoadMark::RoadMarkColor color)
	: DataElement()
	, side_(side)
	, type_(type)
	, width_(width)
	, color_(color)
{
}

void
ParkingSpaceMarking::setParentParkingSpace(ParkingSpace *parkingSpace)
{
	parentParkingSpace_ = parkingSpace;
	addParkingSpaceMarkingChanges(ParkingSpaceMarking::CPSM_ParentChanged);
}

void 
ParkingSpaceMarking::setSide(ParkingSpaceMarkingSide side)
{
	side_ = side;
	addParkingSpaceMarkingChanges(ParkingSpaceMarking::CPSM_SideChanged);
}

void 
ParkingSpaceMarking::setType(LaneRoadMark::RoadMarkType type)
{
	type_ = type;
	addParkingSpaceMarkingChanges(ParkingSpaceMarking::CPSM_TypeChanged);
}

void 
ParkingSpaceMarking::setWidth(double width)
{
	width_ = width;
	addParkingSpaceMarkingChanges(ParkingSpaceMarking::CPSM_WidthChanged);
}

void 
ParkingSpaceMarking::setColor(LaneRoadMark::RoadMarkColor color)
{
	color_ = color;
	addParkingSpaceMarkingChanges(ParkingSpaceMarking::CPSM_ColorChanged);
}

//###################//
// Prototype Pattern //
//###################//

/*! \brief Creates and returns a deep copy clone of this object.
*
*/

ParkingSpaceMarking *
ParkingSpaceMarking::getClone()
{
	// ParkingSpaceMarking //
	//
	ParkingSpaceMarking *clone = new ParkingSpaceMarking(side_, type_, width_, color_);

	return clone;
}

//##################//
// Observer Pattern //
//##################//

/*! \brief Called after all Observers have been notified.
*
* Resets the change flags to 0x0.
*/
void
ParkingSpaceMarking::notificationDone()
{
	markingChanges_ = 0x0;
	DataElement::notificationDone();
}

/*! \brief Add one or more change flags.
*
*/
void
ParkingSpaceMarking::addParkingSpaceMarkingChanges(int changes)
{
	if (changes)
	{
		markingChanges_ |= changes;
		notifyObservers();
	}
}

//###################//
// Visitor Pattern   //
//###################//

/*!
* Accepts a visitor for this section.
*
* \param visitor The visitor that will be visited.
*/
void
ParkingSpaceMarking::accept(Visitor *visitor)
{
	visitor->visit(this);
}




//###################//
// Static Functions  //
//###################//

ParkingSpace::ParkingSpaceAccess
ParkingSpace::parseParkingSpaceAccess(const QString &access)
{
	
	if (access == "all")
		return ParkingSpace::PS_ALL;
	else if (access == "car")
		return ParkingSpace::PS_CAR;
	else if (access == "women")
		return ParkingSpace::PS_WOMEN;
	else if (access == "handicapped")
		return ParkingSpace::PS_HANDICAPPED;
	else if (access == "bus")
		return ParkingSpace::PS_BUS;
	else if (access == "truck")
		return ParkingSpace::PS_TRUCK;
	else if (access == "electric")
		return ParkingSpace::PS_ELECTRIC;
	else if (access == "residents")
		return ParkingSpace::PS_RESIDENTS;
	else
	{
		qDebug("WARNING: unknown access type: %s", access.toUtf8().constData());
		return ParkingSpace::PS_NONE;
	}
}

QString 
ParkingSpace::parseParkingSpaceAccessBack(ParkingSpace::ParkingSpaceAccess access)
{
	if (access == ParkingSpace::PS_NONE)
		return QString("none");
	else if (access == ParkingSpace::PS_ALL)
		return QString("all");
	else if (access == ParkingSpace::PS_CAR)
		return QString("car"); 
	else if (access == ParkingSpace::PS_WOMEN)
		return QString("women");
	else if (access == ParkingSpace::PS_HANDICAPPED)
		return QString("handicapped");
	else if (access == ParkingSpace::PS_BUS)
		return QString("bus");
	else if (access == ParkingSpace::PS_TRUCK)
		return QString("truck");
	else if (access == ParkingSpace::PS_ELECTRIC)
		return QString("electric");
	else if (access == ParkingSpace::PS_RESIDENTS)
		return QString("residents");
	else
	{
		qDebug("WARNING: unknown parking space type.");
		return QString("none");
	}
}

//####################//
// Constructors       //
//####################//

ParkingSpace::ParkingSpace(ParkingSpaceAccess access, const QString &restrictions)
    : DataElement()
    , access_(access)
    , restrictions_(restrictions)
{
}

void 
ParkingSpace::setParentObject(Object *parentObject)
{
	parentObject_ = parentObject;
	addParkingSpaceChanges(ParkingSpace::CPS_ParentChanged);
}

void
ParkingSpace::setAccess(ParkingSpaceAccess access)
{
	access_ = access;
	addParkingSpaceChanges(ParkingSpace::CPS_AccessChanged);
}

void
ParkingSpace::setRestrictions(const QString &restrictions)
{
	restrictions_ = restrictions;
	addParkingSpaceChanges(ParkingSpace::CPS_RestrictionsChanged);
}

void 
ParkingSpace::addMarking(ParkingSpaceMarking *marking)
{
	ParkingSpaceMarking::ParkingSpaceMarkingSide key = marking->getSide();
	if (markingList_.contains(key))
	{
		markingList_.remove(key);
	}

	markingList_.insert(key, marking);
	addParkingSpaceChanges(ParkingSpace::CPS_MarkingsChanged);
   
}


//##################//
// Observer Pattern //
//##################//

/*! \brief Called after all Observers have been notified.
*
* Resets the change flags to 0x0.
*/
void
ParkingSpace::notificationDone()
{
    parkingSpaceChanges_ = 0x0;
    DataElement::notificationDone();
}

/*! \brief Add one or more change flags.
*
*/
void
ParkingSpace::addParkingSpaceChanges(int changes)
{
    if (changes)
    {
        parkingSpaceChanges_ |= changes;
        notifyObservers();
    }
}

//###################//
// Prototype Pattern //
//###################//

/*! \brief Creates and returns a deep copy clone of this object.
*
*/
ParkingSpace *
ParkingSpace::getClone()
{
    // ParkingSpace //
    //
	ParkingSpace *clone = new ParkingSpace(access_, restrictions_);

    return clone;
}

//###################//
// Visitor Pattern   //
//###################//

/*!
* Accepts a visitor for this section.
*
* \param visitor The visitor that will be visited.
*/
void
ParkingSpace::accept(Visitor *visitor)
{
    visitor->visit(this);
}
