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

#include "lanerule.hpp"

#include "lane.hpp"

const QList<QString> LaneRule::KNOWNVALUES {"no stopping at any time", "disabled parking", "car pool"};

//###################//
// Constructors      //
//###################//

LaneRule::LaneRule(double sOffset, const QString &value)
    : DataElement()
    , ruleChanges_(0x0)
    , parentLane_(NULL)
    , sOffset_(sOffset)
    , value_(value)
{
}

LaneRule::~LaneRule()
{
}

void
LaneRule::setParentLane(Lane *parentLane)
{
    parentLane_ = parentLane;
    setParentElement(parentLane);
    addRuleChanges(LaneRule::CLR_ParentLaneChanged);
}

//####################//
// RoadMark Functions //
//####################//

/** Returns the end coordinate of this lane road mark.
* In lane section coordinates [m].
*/
double
LaneRule::getSSectionEnd() const
{
    return parentLane_->getLaneRuleEnd(sOffset_);
}

/** Returns the length coordinate of this lane section.
* In [m].
*/
double
LaneRule::getLength() const
{
    return parentLane_->getLaneRuleEnd(sOffset_) - sOffset_;
}

void
LaneRule::setSOffset(double sOffset)
{
    sOffset_ = sOffset;
    addRuleChanges(LaneRule::CLR_OffsetChanged);
}

void
LaneRule::setValue(const QString &value)
{
    value_ = value;
    addRuleChanges(LaneRule::CLR_ValueChanged);
}


//################//
// OBSERVER       //
//################//

/*! \brief Called after all Observers have been notified.
*
* Resets the change flags to 0x0.
*/
void
LaneRule::notificationDone()
{
    ruleChanges_ = 0x0;
    DataElement::notificationDone();
}

/*! \brief Add one or more change flags.
*
*/
void
LaneRule::addRuleChanges(int changes)
{
    if (changes)
    {
        ruleChanges_ |= changes;
        notifyObservers();
    }
}

//###################//
// Prototype Pattern //
//###################//

/*! \brief Creates and returns a deep copy clone of this object.
*
*/
LaneRule *
LaneRule::getClone()
{
    LaneRule *clone = new LaneRule(sOffset_, value_);

    return clone;
}

//###################//
// Visitor Pattern   //
//###################//

/*! Accepts a visitor for this element.
*/
void
LaneRule::accept(Visitor *visitor)
{
    visitor->visit(this);
}
