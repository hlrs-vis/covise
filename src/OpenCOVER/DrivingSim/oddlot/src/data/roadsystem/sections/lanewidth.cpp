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

#include "lanewidth.hpp"

#include "lanesection.hpp"
#include "lane.hpp"

LaneWidth::LaneWidth(double sOffset, double a, double b, double c, double d)
    : DataElement()
    , Polynomial(a, b, c, d)
    , laneWidthChanges_(0x0)
    , parentLane_(NULL)
    , sOffset_(sOffset)
{
}

LaneWidth::~LaneWidth()
{
}

void
LaneWidth::setParentLane(Lane *parentLane)
{
    parentLane_ = parentLane;
    setParentElement(parentLane);
    addLaneWidthChanges(LaneWidth::CLW_ParentLaneChanged);
}

/*! \brief Convenience function. Calls f(sSection) of the Polynomial class.
*/
double
LaneWidth::getWidth(double sSection) const
{
    return f(sSection - sOffset_);
}

double
LaneWidth::getSlope(double sSection) const
{
    return df(sSection - sOffset_);
}

double
LaneWidth::getCurvature(double sSection) const
{
    return ddf(sSection - sOffset_);
}

//####################//
// Width Functions //
//####################//

/** Returns the end coordinate of this lane road mark.
* In lane section coordinates [m].
*/
double
LaneWidth::getSSectionStartAbs() const
{
    return sOffset_ + getParentLane()->getParentLaneSection()->getSStart();
}

/** Returns the end coordinate of this lane road mark.
* In lane section coordinates [m].
*/
double
LaneWidth::getSSectionEnd() const
{
    return parentLane_->getWidthEnd(sOffset_);
}

/** Returns the length coordinate of this lane section.
* In [m].
*/
double
LaneWidth::getLength() const
{
	return parentLane_->getWidthEnd(sOffset_) - getSSectionStartAbs();
}

void
LaneWidth::setSOffset(double sOffset)
{
    sOffset_ = sOffset;
    addLaneWidthChanges(LaneWidth::CLW_OffsetChanged);
}

//################//
// OBSERVER       //
//################//

/*! \brief Called after all Observers have been notified.
*
* Resets the change flags to 0x0.
*/
void
LaneWidth::notificationDone()
{
    laneWidthChanges_ = 0x0;
    DataElement::notificationDone();
}

/*! \brief Add one or more change flags.
*
*/
void
LaneWidth::addLaneWidthChanges(int changes)
{
    if (changes)
    {
        laneWidthChanges_ |= changes;
        notifyObservers();
    }
    if (parentLane_)
        parentLane_->addLaneChanges(Lane::CLN_WidthsChanged);
}

//###################//
// Prototype Pattern //
//###################//

/*! \brief Creates and returns a deep copy clone of this object.
*
*/
LaneWidth *
LaneWidth::getClone()
{
    LaneWidth *clone = new LaneWidth(sOffset_, a_, b_, c_, d_);

    return clone;
}

void
LaneWidth::setParameters(double a, double b, double c, double d)
{
    protectedSetParameters(a, b, c, d);

    addLaneWidthChanges(LaneWidth::CLW_WidthChanged);
}

//###################//
// Visitor Pattern   //
//###################//

/*! Accepts a visitor for this element.
*/
void
LaneWidth::accept(Visitor *visitor)
{
    visitor->visit(this);
}
