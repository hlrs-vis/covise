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

#include "lanespeed.hpp"

#include "lane.hpp"

LaneSpeed::LaneSpeed(double sOffset, double max = -1.0)
    : DataElement()
    , laneSpeedChanges_(0x0)
    , parentLane_(NULL)
    , sOffset_(sOffset)
    , max_(max)
{
}

LaneSpeed::~LaneSpeed()
{
}

void
LaneSpeed::setParentLane(Lane *parentLane)
{
    parentLane_ = parentLane;
    setParentElement(parentLane);
    addLaneSpeedChanges(LaneSpeed::CLS_ParentLaneChanged);
}

//####################//
// Speed Functions //
//####################//

/** Returns the end coordinate of this lane road mark.
* In lane section coordinates [m].
*/
double
LaneSpeed::getSSectionEnd() const
{
    return parentLane_->getSpeedEnd(sOffset_);
}

/** Returns the length coordinate of this lane section.
* In [m].
*/
double
LaneSpeed::getLength() const
{
    return parentLane_->getSpeedEnd(sOffset_) - sOffset_;
}

void
LaneSpeed::setSOffset(double sOffset)
{
    sOffset_ = sOffset;
    addLaneSpeedChanges(LaneSpeed::CLS_OffsetChanged);
}

void
LaneSpeed::setMaxSpeed(double max)
{
    max_ = max;
    addLaneSpeedChanges(LaneSpeed::CLS_MaxSpeedChanged);
}

//################//
// OBSERVER       //
//################//

/*! \brief Called after all Observers have been notified.
*
* Resets the change flags to 0x0.
*/
void
LaneSpeed::notificationDone()
{
    laneSpeedChanges_ = 0x0;
    DataElement::notificationDone();
}

/*! \brief Add one or more change flags.
*
*/
void
LaneSpeed::addLaneSpeedChanges(int changes)
{
    if (changes)
    {
        laneSpeedChanges_ |= changes;
        notifyObservers();
    }
}

//###################//
// Prototype Pattern //
//###################//

/*! \brief Creates and returns a deep copy clone of this object.
*
*/
LaneSpeed *
LaneSpeed::getClone()
{
    LaneSpeed *clone = new LaneSpeed(sOffset_, max_);

    return clone;
}

//###################//
// Visitor Pattern   //
//###################//

/*! Accepts a visitor for this element.
*/
void
LaneSpeed::accept(Visitor *visitor)
{
    visitor->visit(this);
}
