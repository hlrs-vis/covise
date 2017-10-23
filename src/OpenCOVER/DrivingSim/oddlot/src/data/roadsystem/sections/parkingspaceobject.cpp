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
#include "objectobject.hpp"


//###################//
// Static Functions  //
//###################//

ParkingSpace::ParkingSpaceMarking::ParkingSpaceMarkingSide
ParkingSpace::ParkingSpaceMarking::parseParkingSpaceMarkingSide(const QString &side)
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
ParkingSpace::ParkingSpaceMarking::parseParkingSpaceMarkingSideBack(ParkingSpaceMarking::ParkingSpaceMarkingSide side)
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

ParkingSpace::ParkingSpaceMarking::ParkingSpaceMarking(Object *parentObject, ParkingSpaceMarkingSide side, LaneRoadMark::RoadMarkType type, double width, LaneRoadMark::RoadMarkColor color)
	: parentObject_(parentObject)
	, side_(side)
	, type_(type)
	, width_(width)
	, color_(color)
{
}

void
ParkingSpace::ParkingSpaceMarking::setParentParkingSpace(ParkingSpace *parkingSpace)
{
	parentParkingSpace_ = parkingSpace;
	parentObject_->addObjectChanges(Object::CEL_ParkingSpaceChange);
}

void 
ParkingSpace::ParkingSpaceMarking::setSide(ParkingSpaceMarkingSide side)
{
	side_ = side;
	parentObject_->addObjectChanges(Object::CEL_ParkingSpaceChange);
}

void 
ParkingSpace::ParkingSpaceMarking::setType(LaneRoadMark::RoadMarkType type)
{
	type_ = type;
	parentObject_->addObjectChanges(Object::CEL_ParkingSpaceChange);
}

void 
ParkingSpace::ParkingSpaceMarking::setWidth(double width)
{
	width_ = width;
	parentObject_->addObjectChanges(Object::CEL_ParkingSpaceChange);
}

void 
ParkingSpace::ParkingSpaceMarking::setColor(LaneRoadMark::RoadMarkColor color)
{
	color_ = color;
	parentObject_->addObjectChanges(Object::CEL_ParkingSpaceChange);
}

//###################//
// Prototype Pattern //
//###################//

/*! \brief Creates and returns a deep copy clone of this object.
*
*/

ParkingSpace::ParkingSpaceMarking *
ParkingSpace::ParkingSpaceMarking::getClone()
{
	// ParkingSpaceMarking //
	//
	ParkingSpaceMarking *clone = new ParkingSpaceMarking(parentObject_, side_, type_, width_, color_);

	return clone;
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

ParkingSpace::ParkingSpace(Object *parentObject, ParkingSpaceAccess access, const QString &restrictions)
	: parentObject_(parentObject)
    , access_(access)
    , restrictions_(restrictions)
{
}

void
ParkingSpace::setParentObject(Object *parentObject)
{
	parentObject_ = parentObject;
	parentObject_->addObjectChanges(Object::CEL_ParkingSpaceChange);
}

void
ParkingSpace::setAccess(ParkingSpaceAccess access)
{
	access_ = access;
	parentObject_->addObjectChanges(Object::CEL_ParkingSpaceChange);
}

void
ParkingSpace::setRestrictions(const QString &restrictions)
{
	restrictions_ = restrictions;
	parentObject_->addObjectChanges(Object::CEL_ParkingSpaceChange);
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
	parentObject_->addObjectChanges(Object::CEL_ParkingSpaceChange);
   
}

bool 
ParkingSpace::addMarking(QString side, QString type, double width, QString color)
{
	ParkingSpace::ParkingSpaceMarking::ParkingSpaceMarkingSide parkSide = ParkingSpace::ParkingSpaceMarking::parseParkingSpaceMarkingSide(side);
	if (parkSide == ParkingSpace::ParkingSpaceMarking::PSM_NONE)
	{
		return false;
	}
	else
	{
		ParkingSpaceMarking *marking = new ParkingSpaceMarking(parentObject_, parkSide, LaneRoadMark::parseRoadMarkType(type), width, LaneRoadMark::parseRoadMarkColor(color));		
		addMarking(marking);

		return true;
	}
}

bool 
ParkingSpace::getMarking(int i, QString &side, QString &type, double &width, QString &color)
{
	if (i >= markingList_.size())
	{
		return false;
	}

	QMap<ParkingSpaceMarking::ParkingSpaceMarkingSide, ParkingSpaceMarking *>::const_iterator it = markingList_.constBegin() + i;
	ParkingSpaceMarking *marking = it.value();
	side = ParkingSpaceMarking::parseParkingSpaceMarkingSideBack(marking->getSide());
	type = LaneRoadMark::parseRoadMarkTypeBack(marking->getType());
	width = marking->getWidth();
	color = LaneRoadMark::parseRoadMarkColorBack(marking->getColor());

	return true;
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
	ParkingSpace *clone = new ParkingSpace(parentObject_, access_, restrictions_);

    return clone;
}

