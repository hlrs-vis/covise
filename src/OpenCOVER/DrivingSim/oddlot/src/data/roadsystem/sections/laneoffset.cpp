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

#include "laneoffset.hpp"

#include "../rsystemelementroad.hpp"

LaneOffset::LaneOffset(double sOffset, double a, double b, double c, double d)
    : DataElement()
    , Polynomial(a, b, c, d)
    , laneOffsetChanges_(0x0)
    , parentRoad_(NULL)
    , sOffset_(sOffset)
{
}

LaneOffset::~LaneOffset()
{
}

void
LaneOffset::setParentRoad(RSystemElementRoad *parentRoad)
{
	parentRoad_ = parentRoad;
    setParentElement(parentRoad);
    addLaneOffsetChanges(LaneOffset::CLO_ParentRoadChanged);
}

/*! \brief Convenience function. Calls f(sSection) of the Polynomial class.
*/
double
LaneOffset::getOffset(double sSection) const
{
    return f(sSection - sOffset_);
}

double
LaneOffset::getSlope(double sSection) const
{
    return df(sSection - sOffset_);
}

double
LaneOffset::getCurvature(double sSection) const
{
    return ddf(sSection - sOffset_);
}

void
LaneOffset::setSOffset(double sOffset)
{
	sOffset_ = sOffset;
	addLaneOffsetChanges(LaneOffset::CLO_SOffsetChanged);
}

//################//
// OBSERVER       //
//################//

/*! \brief Called after all Observers have been notified.
*
* Resets the change flags to 0x0.
*/
void
LaneOffset::notificationDone()
{
    laneOffsetChanges_ = 0x0;
    DataElement::notificationDone();
}

/*! \brief Add one or more change flags.
*
*/
void
LaneOffset::addLaneOffsetChanges(int changes)
{
    if (changes)
    {
        laneOffsetChanges_ |= changes;
        notifyObservers();
    }
    if (parentRoad_)
        parentRoad_->addRoadChanges(RSystemElementRoad::CRD_LaneOffsetChange);
}

//###################//
// Prototype Pattern //
//###################//

/*! \brief Creates and returns a deep copy clone of this object.
*
*/
LaneOffset *
LaneOffset::getClone()
{
    LaneOffset *clone = new LaneOffset(sOffset_, a_, b_, c_, d_);

    return clone;
}

void
LaneOffset::setParameters(double a, double b, double c, double d)
{
    protectedSetParameters(a, b, c, d);

    addLaneOffsetChanges(LaneOffset::CLO_OffsetChanged);
}

//###################//
// Visitor Pattern   //
//###################//

/*! Accepts a visitor for this element.
*/
void
LaneOffset::accept(Visitor *visitor)
{
    visitor->visit(this);
}
