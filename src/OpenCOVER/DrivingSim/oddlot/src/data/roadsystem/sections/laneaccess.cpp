/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   25.02.2010
**
**************************************************************************/

#include "laneaccess.hpp"

#include "lane.hpp"

LaneAccess::LaneRestriction 
LaneAccess::parseLaneRestriction(const QString &restriction)
{
	if (restriction == "simulator")
	{
		return LaneAccess::LAR_SIMULATOR;
	}
	else if (restriction == "autonomous traffic")
	{
		return LaneAccess::LAR_AUTONOMOUS;
	}
	else if (restriction == "pedestrian")
	{
		return LaneAccess::LAR_PEDESTRIAN;
	}
	else if (restriction == "none" )
	{
		return LaneAccess::LAR_NONE;
	}
	else
	{
		qDebug("WARNING: unknown lane restriction type: %s", restriction.toUtf8().constData());
		return LaneAccess::LAR_UNKNOWN;
	}
}

QString 
LaneAccess::parseLaneRestrictionBack(LaneAccess::LaneRestriction restriction)
{
	if (restriction == LaneAccess::LAR_SIMULATOR)
	{
		return QString("simulator");
	}
	else if (restriction == LaneAccess::LAR_AUTONOMOUS)
	{
		return QString("autonomous traffic");
	}
	else if (restriction == LaneAccess::LAR_PEDESTRIAN)
	{
		return QString("pedestrian");
	}
	else if (restriction == LaneAccess::LAR_NONE)
	{
		return QString("none");
	}
	else
	{
		qDebug("WARNING: unknown lane restriction type");
		return QString("unknown");
	}
}


//###################//
// Constructors      //
//###################//

LaneAccess::LaneAccess(double sOffset, LaneRestriction restriction)
    : DataElement()
    , accessChanges_(0x0)
    , parentLane_(NULL)
    , sOffset_(sOffset)
    , restriction_(restriction)
{
}

LaneAccess::~LaneAccess()
{
}

void
LaneAccess::setParentLane(Lane *parentLane)
{
    parentLane_ = parentLane;
    setParentElement(parentLane);
    addAccessChanges(LaneAccess::CLA_ParentLaneChanged);
}

//####################//
// RoadMark Functions //
//####################//

/** Returns the end coordinate of this lane road mark.
* In lane section coordinates [m].
*/
double
LaneAccess::getSSectionEnd() const
{
    return parentLane_->getLaneAccessEnd(sOffset_);
}

/** Returns the length coordinate of this lane section.
* In [m].
*/
double
LaneAccess::getLength() const
{
    return parentLane_->getLaneAccessEnd(sOffset_) - sOffset_;
}

void
LaneAccess::setSOffset(double sOffset)
{
    sOffset_ = sOffset;
    addAccessChanges(LaneAccess::CLA_OffsetChanged);
}

void
LaneAccess::setRestriction(LaneRestriction restriction)
{
	restriction_ = restriction;
    addAccessChanges(LaneAccess::CLA_RestrictionChanged);
}


//################//
// OBSERVER       //
//################//

/*! \brief Called after all Observers have been notified.
*
* Resets the change flags to 0x0.
*/
void
LaneAccess::notificationDone()
{
    accessChanges_ = 0x0;
    DataElement::notificationDone();
}

/*! \brief Add one or more change flags.
*
*/
void
LaneAccess::addAccessChanges(int changes)
{
    if (changes)
    {
        accessChanges_ |= changes;
        notifyObservers();
    }
}

//###################//
// Prototype Pattern //
//###################//

/*! \brief Creates and returns a deep copy clone of this object.
*
*/
LaneAccess *
LaneAccess::getClone()
{
    LaneAccess *clone = new LaneAccess(sOffset_, restriction_);

    return clone;
}

//###################//
// Visitor Pattern   //
//###################//

/*! Accepts a visitor for this element.
*/
void
LaneAccess::accept(Visitor *visitor)
{
    visitor->visit(this);
}
