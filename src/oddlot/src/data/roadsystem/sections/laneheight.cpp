/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   23.02.2010
**
**************************************************************************/

#include "laneheight.hpp"

#include "lanesection.hpp"
#include "lane.hpp"

LaneHeight::LaneHeight(double sOffset, double inner, double outer)
    : DataElement()
    , laneHeightChanges_(0x0)
    , parentLane_(NULL)
    , sOffset_(sOffset)
    , inner_(inner)
    , outer_(outer)
{
}

LaneHeight::~LaneHeight()
{
}

void
LaneHeight::setParentLane(Lane *parentLane)
{
    parentLane_ = parentLane;
    setParentElement(parentLane);
    addLaneHeightChanges(LaneHeight::CLW_ParentLaneChanged);
}

/*! \brief Convenience function
*/
double
LaneHeight::getInnerHeight() const
{
    return inner_;
}
double
LaneHeight::getOuterHeight() const
{
    return outer_;
}

//####################//
// Height Functions //
//####################//

/** Returns the end coordinate of this lane road mark.
* In lane section coordinates [m].
*/
double
LaneHeight::getSSectionStartAbs() const
{
    return sOffset_ + getParentLane()->getParentLaneSection()->getSStart();
}

/** Returns the end coordinate of this lane road mark.
* In lane section coordinates [m].
*/
double
LaneHeight::getSSectionEnd() const
{
    return parentLane_->getHeightEnd(sOffset_);
}

/** Returns the length coordinate of this lane section.
* In [m].
*/
double
LaneHeight::getLength() const
{
    return parentLane_->getHeightEnd(sOffset_) - sOffset_;
}

void
LaneHeight::setSOffset(double sOffset)
{
    sOffset_ = sOffset;
    addLaneHeightChanges(LaneHeight::CLW_OffsetChanged);
}

//################//
// OBSERVER       //
//################//

/*! \brief Called after all Observers have been notified.
*
* Resets the change flags to 0x0.
*/
void
LaneHeight::notificationDone()
{
    laneHeightChanges_ = 0x0;
    DataElement::notificationDone();
}

/*! \brief Add one or more change flags.
*
*/
void
LaneHeight::addLaneHeightChanges(int changes)
{
    if (changes)
    {
        laneHeightChanges_ |= changes;
        notifyObservers();
    }
}

//###################//
// Prototype Pattern //
//###################//

/*! \brief Creates and returns a deep copy clone of this object.
*
*/
LaneHeight *
LaneHeight::getClone()
{
    LaneHeight *clone = new LaneHeight(sOffset_, inner_, outer_);

    return clone;
}

//###################//
// Visitor Pattern   //
//###################//

/*! Accepts a visitor for this element.
*/
void
LaneHeight::accept(Visitor *visitor)
{
    visitor->visit(this);
}
