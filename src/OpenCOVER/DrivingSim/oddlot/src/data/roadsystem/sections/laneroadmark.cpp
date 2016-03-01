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

#include "laneroadmark.hpp"

#include "lane.hpp"

//###################//
// Static Functions  //
//###################//

LaneRoadMark::RoadMarkType
LaneRoadMark::parseRoadMarkType(const QString &type)
{
    if (type == "none")
        return LaneRoadMark::RMT_NONE;
    else if (type == "solid")
        return LaneRoadMark::RMT_SOLID;
    else if (type == "broken")
        return LaneRoadMark::RMT_BROKEN;
    else if (type == "solid solid")
        return LaneRoadMark::RMT_SOLID_SOLID;
    else if (type == "solid broken")
        return LaneRoadMark::RMT_SOLID_BROKEN;
    else if (type == "broken solid")
        return LaneRoadMark::RMT_BROKEN_SOLID;
    else
    {
        qDebug("WARNING: unknown road mark type: %s", type.toUtf8().constData());
        return LaneRoadMark::RMT_NONE;
    }
};

QString
LaneRoadMark::parseRoadMarkTypeBack(LaneRoadMark::RoadMarkType type)
{
    if (type == LaneRoadMark::RMT_NONE)
        return QString("none");
    else if (type == LaneRoadMark::RMT_SOLID)
        return QString("solid");
    else if (type == LaneRoadMark::RMT_BROKEN)
        return QString("broken");
    else if (type == LaneRoadMark::RMT_SOLID_SOLID)
        return QString("solid solid");
    else if (type == LaneRoadMark::RMT_SOLID_BROKEN)
        return QString("solid broken");
    else if (type == LaneRoadMark::RMT_BROKEN_SOLID)
        return QString("broken solid");
    else
    {
        qDebug("WARNING: unknown road mark type.");
        return QString("none");
    }
};

LaneRoadMark::RoadMarkWeight
LaneRoadMark::parseRoadMarkWeight(const QString &type)
{
    if (type == "standard")
        return LaneRoadMark::RMW_STANDARD;
    else if (type == "bold")
        return LaneRoadMark::RMW_BOLD;
    else
    {
        qDebug("WARNING: unknown road mark weight type: %s", type.toUtf8().constData());
        return LaneRoadMark::RMW_STANDARD;
    }
};

QString
LaneRoadMark::parseRoadMarkWeightBack(LaneRoadMark::RoadMarkWeight type)
{
    if (type == LaneRoadMark::RMW_STANDARD)
        return QString("standard");
    else if (type == LaneRoadMark::RMW_BOLD)
        return QString("bold");
    else
    {
        qDebug("WARNING: unknown road mark weight type.");
        return "standard";
    }
};

LaneRoadMark::RoadMarkColor
LaneRoadMark::parseRoadMarkColor(const QString &type)
{
    if (type == "standard")
        return LaneRoadMark::RMC_STANDARD;
    else if (type == "yellow")
        return LaneRoadMark::RMC_YELLOW;
    else
    {
        qDebug("WARNING: unknown road mark color type: %s", type.toUtf8().constData());
        return LaneRoadMark::RMC_STANDARD;
    }
};

QString
LaneRoadMark::parseRoadMarkColorBack(LaneRoadMark::RoadMarkColor type)
{
    if (type == LaneRoadMark::RMC_STANDARD)
        return "standard";
    else if (type == LaneRoadMark::RMC_YELLOW)
        return "yellow";
    else
    {
        qDebug("WARNING: unknown road mark color type");
        return "standard";
    }
};

LaneRoadMark::RoadMarkLaneChange
LaneRoadMark::parseRoadMarkLaneChange(const QString &type)
{
    if (type == "increase")
        return LaneRoadMark::RMLC_INCREASE;
    else if (type == "decrease")
        return LaneRoadMark::RMLC_DECREASE;
    else if (type == "both")
        return LaneRoadMark::RMLC_BOTH;
    else if (type == "none")
        return LaneRoadMark::RMLC_NONE;
    else
    {
        qDebug("WARNING: unknown road mark lane change type: %s", type.toUtf8().constData());
        return LaneRoadMark::RMLC_BOTH;
    }
}

QString
LaneRoadMark::parseRoadMarkLaneChangeBack(LaneRoadMark::RoadMarkLaneChange type)
{
    if (type == LaneRoadMark::RMLC_INCREASE)
        return "increase";
    else if (type == LaneRoadMark::RMLC_DECREASE)
        return "decrease";
    else if (type == LaneRoadMark::RMLC_BOTH)
        return "both";
    else if (type == LaneRoadMark::RMLC_NONE)
        return "none";
    else
    {
        qDebug("WARNING: unknown road mark lane change type");
        return "none";
    }
}

//###################//
// Constructors      //
//###################//

LaneRoadMark::LaneRoadMark(double sOffset, RoadMarkType type, RoadMarkWeight weight, RoadMarkColor color, double width, RoadMarkLaneChange laneChange)
    : DataElement()
    , roadMarkChanges_(0x0)
    , parentLane_(NULL)
    , sOffset_(sOffset)
    , type_(type)
    , weight_(weight)
    , color_(color)
    , width_(width)
    , laneChange_(laneChange)
{
}

LaneRoadMark::~LaneRoadMark()
{
}

void
LaneRoadMark::setParentLane(Lane *parentLane)
{
    parentLane_ = parentLane;
    setParentElement(parentLane);
    addRoadMarkChanges(LaneRoadMark::CLR_ParentLaneChanged);
}

//####################//
// RoadMark Functions //
//####################//

/** Returns the end coordinate of this lane road mark.
* In lane section coordinates [m].
*/
double
LaneRoadMark::getSSectionEnd() const
{
    return parentLane_->getRoadMarkEnd(sOffset_);
}

/** Returns the length coordinate of this lane section.
* In [m].
*/
double
LaneRoadMark::getLength() const
{
    return parentLane_->getRoadMarkEnd(sOffset_) - sOffset_;
}

void
LaneRoadMark::setSOffset(double sOffset)
{
    sOffset_ = sOffset;
    addRoadMarkChanges(LaneRoadMark::CLR_OffsetChanged);
}

void
LaneRoadMark::setRoadMarkType(RoadMarkType type)
{
    type_ = type;
    addRoadMarkChanges(LaneRoadMark::CLR_TypeChanged);
}

void
LaneRoadMark::setRoadMarkWeight(RoadMarkWeight weight)
{
    weight_ = weight;
    addRoadMarkChanges(LaneRoadMark::CLR_WeightChanged);
}

void
LaneRoadMark::setRoadMarkColor(RoadMarkColor color)
{
    color_ = color;
    addRoadMarkChanges(LaneRoadMark::CLR_ColorChanged);
}

void
LaneRoadMark::setRoadMarkWidth(double width)
{
    width_ = width;
    addRoadMarkChanges(LaneRoadMark::CLR_WidthChanged);
}

void
LaneRoadMark::setRoadMarkLaneChange(RoadMarkLaneChange permission)
{
    laneChange_ = permission;
    addRoadMarkChanges(LaneRoadMark::CLR_LaneChangeChanged);
}

//################//
// OBSERVER       //
//################//

/*! \brief Called after all Observers have been notified.
*
* Resets the change flags to 0x0.
*/
void
LaneRoadMark::notificationDone()
{
    roadMarkChanges_ = 0x0;
    DataElement::notificationDone();
}

/*! \brief Add one or more change flags.
*
*/
void
LaneRoadMark::addRoadMarkChanges(int changes)
{
    if (changes)
    {
        roadMarkChanges_ |= changes;
        notifyObservers();
    }
}

//###################//
// Prototype Pattern //
//###################//

/*! \brief Creates and returns a deep copy clone of this object.
*
*/
LaneRoadMark *
LaneRoadMark::getClone()
{
    LaneRoadMark *clone = new LaneRoadMark(sOffset_, type_, weight_, color_, width_, laneChange_);

    return clone;
}

//###################//
// Visitor Pattern   //
//###################//

/*! Accepts a visitor for this element.
*/
void
LaneRoadMark::accept(Visitor *visitor)
{
    visitor->visit(this);
}
