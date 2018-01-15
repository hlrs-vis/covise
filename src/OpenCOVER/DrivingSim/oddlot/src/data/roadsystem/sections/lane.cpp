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

#include "lane.hpp"

#include "lanesection.hpp"

#include "lanewidth.hpp"
#include "laneroadmark.hpp"
#include "lanespeed.hpp"
#include "laneheight.hpp"
#include "lanerule.hpp"
#include "laneaccess.hpp"

#include "src/data/roadsystem/rsystemelementroad.hpp"

#include "math.h"

//###################//
// Static Functions  //
//###################//

const int Lane::NOLANE = -99;

Lane::LaneType
Lane::parseLaneType(const QString &type)
{
    if (type == "none")
        return Lane::LT_NONE;
    else if (type == "driving")
        return Lane::LT_DRIVING;
    else if (type == "stop")
        return Lane::LT_STOP;
    else if (type == "shoulder")
        return Lane::LT_SHOULDER;
    else if (type == "biking")
        return Lane::LT_BIKING;
    else if (type == "sidewalk")
        return Lane::LT_SIDEWALK;
    else if (type == "border")
        return Lane::LT_BORDER;
    else if (type == "restricted")
        return Lane::LT_RESTRICTED;
    else if (type == "parking")
        return Lane::LT_PARKING;
	else if (type == "bidirectional")
		return Lane::LT_BIDIRECTIONAL;
	else if (type == "median")
		return Lane::LT_MEDIAN;
    else if (type == "mwyEntry")
        return Lane::LT_MWYENTRY;
    else if (type == "mwyExit")
        return Lane::LT_MWYEXIT;
    else if (type == "special1")
        return Lane::LT_SPECIAL1;
    else if (type == "special2")
        return Lane::LT_SPECIAL2;
    else if (type == "special3")
        return Lane::LT_SPECIAL3;
	else if (type == "roadWorks")
		return Lane::LT_ROADWORKS;
	else if (type == "tram")
		return Lane::LT_TRAM;
	else if (type == "rail")
		return Lane::LT_RAIL;
	else if (type == "entry")
		return Lane::LT_ENTRY;
	else if (type == "exit")
		return Lane::LT_EXIT;
	else if (type == "offRamp")
		return Lane::LT_OFFRAMP;
	else if (type == "onRamp")
		return Lane::LT_ONRAMP;
    else
    {
        qDebug("WARNING: unknown lane type: %s", type.toUtf8().constData());
        return Lane::LT_NONE;
    }
}

QString
Lane::parseLaneTypeBack(Lane::LaneType type)
{
    if (type == Lane::LT_NONE)
        return QString("none");
    else if (type == Lane::LT_DRIVING)
        return QString("driving");
    else if (type == Lane::LT_STOP)
        return QString("stop");
    else if (type == Lane::LT_SHOULDER)
        return QString("shoulder");
    else if (type == Lane::LT_BIKING)
        return QString("biking");
    else if (type == Lane::LT_SIDEWALK)
        return QString("sidewalk");
    else if (type == Lane::LT_BORDER)
        return QString("border");
    else if (type == Lane::LT_RESTRICTED)
        return QString("restricted");
    else if (type == Lane::LT_PARKING)
        return QString("parking");
	else if (type == Lane::LT_BIDIRECTIONAL)
		return QString("bidirectional");
	else if (type == Lane::LT_MEDIAN)
		return QString("median");
    else if (type == Lane::LT_MWYENTRY)
        return QString("mwyEntry");
    else if (type == Lane::LT_MWYEXIT)
        return QString("mwyExit");
    else if (type == Lane::LT_SPECIAL1)
        return QString("special1");
    else if (type == Lane::LT_SPECIAL2)
        return QString("special2");
    else if (type == Lane::LT_SPECIAL3)
        return QString("special3");
	else if (type == Lane::LT_ROADWORKS)
		return QString("roadWorks");
	else if (type == Lane::LT_TRAM)
		return QString("tram");
	else if (type == Lane::LT_RAIL)
		return QString("rail");
	else if (type == Lane::LT_ENTRY)
		return QString("entry");
	else if (type == Lane::LT_EXIT)
		return QString("exit");
	else if (type == Lane::LT_OFFRAMP)
		return QString("offRamp");
	else if (type == Lane::LT_ONRAMP)
		return QString("onRamp");
    else
    {
        qDebug("WARNING: unknown lane type.");
        return QString("none");
    }
}

//###################//
// Constructors      //
//###################//

Lane::Lane(int id, Lane::LaneType type, bool level, int predecessorId, int successorId)
    : DataElement()
    , laneChanges_(0x0)
    , parentLaneSection_(NULL)
    , id_(id)
    , type_(type)
    , level_(level)
    , predecessorId_(predecessorId)
    , successorId_(successorId)
{
}

Lane::~Lane()
{
    // Delete child nodes //
    //
    foreach (LaneWidth *child, widths_)
        delete child;

    foreach (LaneRoadMark *child, marks_)
        delete child;

    foreach (LaneSpeed *child, speeds_)
        delete child;

	foreach(LaneRule *child, rules_)
		delete child;
}

void
Lane::setId(int id)
{
    if (parentLaneSection_)
    {
        QMap<int, Lane *> lanes = parentLaneSection_->getLanes();
        if (lanes.contains(id))
        {
            qDebug() << "WARNING: Trying to get lane width but coordinate is out of bounds! Number of width entries in lane " << id_ << ": " << widths_.size() << ", laneSection at " << getParentLaneSection()->getSStart() << ", road " << getParentLaneSection()->getParentRoad()->getID();
            return;
        }
        LaneSection *parentLaneSection = parentLaneSection_;
        parentLaneSection_->removeLane(this);
        id_ = id;
        parentLaneSection->addLane(this);
    }
    else
    {
        id_ = id;
    }
    addLaneChanges(Lane::CLN_IdChanged);
}

void
Lane::setLaneType(Lane::LaneType laneType)
{
    type_ = laneType;
    addLaneChanges(Lane::CLN_TypeChanged);
}

void
Lane::setLevel(bool level)
{
    level_ = level;
    addLaneChanges(Lane::CLN_LevelChanged);
}

void
Lane::setPredecessor(int id)
{
    predecessorId_ = id;
    addLaneChanges(Lane::CLN_PredecessorChanged);
}

void
Lane::setSuccessor(int id)
{
    successorId_ = id;
    addLaneChanges(Lane::CLN_SuccessorChanged);
}

void
Lane::setParentLaneSection(LaneSection *parentLaneSection)
{
    parentLaneSection_ = parentLaneSection;
    setParentElement(parentLaneSection);
    addLaneChanges(Lane::CLN_ParentLaneSectionChanged);
}

//###################//
// Width Functions   //
//###################//

/** Adds a lane width entry to this lane.
*/
void
Lane::addWidthEntry(LaneWidth *widthEntry)
{
    widths_.insert(widthEntry->getSOffset(), widthEntry);
    widthEntry->setParentLane(this);
    addLaneChanges(Lane::CLN_WidthsChanged);
}

bool Lane::delWidthEntry(LaneWidth *widthEntry)
{
    if (widths_.remove(widthEntry->getSSectionStart()) == 0)
        return false;
    addLaneChanges(Lane::CLN_WidthsChanged);
    return true;
}

LaneWidth *Lane::getWidthEntry(double sSection) const
{
    return (widths_.value(sSection, NULL));
}
bool
Lane::moveWidthEntry(double oldS, double newS)
{
    // Section //
    //
    LaneWidth *section = widths_.value(oldS, NULL);
    if (!section)
    {
        return false;
    }

    // Previous section //
    //
    double previousS = 0.0;
    if (newS > section->getSSectionStart())
    {
        // Expand previous section //
        //
        previousS = section->getSSectionStart() - 0.001;
    }
    else
    {
        // Shrink previous section //
        //
        previousS = newS;
    }
    LaneWidth *previousSection = getWidthEntry(previousS);
    if (previousSection)
    {
        previousSection->addLaneWidthChanges(LaneWidth::CLW_OffsetChanged);
    }

    // Set and insert //
    //
    section->setSOffset(newS);
    widths_.remove(oldS);
    widths_.insert(newS, section);

    addLaneChanges(Lane::CLN_WidthsChanged);

    return true;
}

/** Returns the width of this lane at the given coordinate sSection.
*/
double
Lane::getWidth(double sSection) const
{
    QMap<double, LaneWidth *>::const_iterator i = widths_.upperBound(sSection);
    if (i == widths_.constBegin())
    {
        qDebug() << "WARNING: Trying to get lane width but coordinate is out of bounds! Number of width entries in lane " << id_ << ": " << widths_.size() << ", laneSection at " << getParentLaneSection()->getSStart() << ", road " << getParentLaneSection()->getParentRoad()->getID();
        return 0.0;
    }
    else
    {
        --i;
        return i.value()->getWidth(sSection);
    }
}

/** Returns the slope of this lane at the given coordinate sSection.
*/
double
Lane::getSlope(double sSection) const
{
    QMap<double, LaneWidth *>::const_iterator i = widths_.upperBound(sSection);
    if (i == widths_.constBegin())
    {
        qDebug() << "WARNING: Trying to get lane slope but coordinate is out of bounds! Number of width entries in lane " << id_ << ": " << widths_.size() << ", laneSection at " << getParentLaneSection()->getSStart() << ", road " << getParentLaneSection()->getParentRoad()->getID();
        return 0.0;
    }
    else
    {
        --i;
        return i.value()->getSlope(sSection);
    }
}

/** Returns the absolute end coordinate of the width entry
* containing the lane section coordinate sSection.
* In lane section coordinates [m].
*/
double
Lane::getWidthEnd(double sSection) const
{
    QMap<double, LaneWidth *>::const_iterator nextIt = widths_.upperBound(sSection);
    if (nextIt == widths_.constEnd())
    {
        return parentLaneSection_->getSEnd(); // lane section: [0.0, length]
    }
    else
    {
        return (*nextIt)->getSSectionStart() + parentLaneSection_->getSStart();
    }
}

//####################//
// RoadMark Functions //
//####################//

/** Adds a road mark entry to this lane.
*/
void
Lane::addRoadMarkEntry(LaneRoadMark *roadMarkEntry)
{
    marks_.insert(roadMarkEntry->getSSectionStart(), roadMarkEntry);
    roadMarkEntry->setParentLane(this);
    addLaneChanges(Lane::CLN_RoadMarksChanged);
}

/** Returns the road mark entry at the given section coordinate (sSection).
*/
LaneRoadMark *
Lane::getRoadMarkEntry(double sSection) const
{
    QMap<double, LaneRoadMark *>::const_iterator i = marks_.upperBound(sSection);
    if (i == marks_.constBegin())
    {
        qDebug("WARNING: Trying to get lane road mark but coordinate is out of bounds!");
        return NULL;
    }
    else
    {
        --i;
        return i.value();
    }
}

/** Returns the end coordinate of the road mark
* containing the lane section coordinate sSection.
* In lane section coordinates [m].
* If the road mark is the last in the list, the end of
* the lane section will be returned. Otherwise the end of
* a road mark is the start coordinate of the successing one.
*/
double
Lane::getRoadMarkEnd(double sSection) const
{
    //sSection = 0.0; // ???
    QMap<double, LaneRoadMark *>::const_iterator nextIt = marks_.upperBound(sSection);
    if (nextIt == marks_.constEnd())
    {
        return parentLaneSection_->getLength(); // lane section: [0.0, length]
    }
    else
    {
        return (*nextIt)->getSSectionStart();
    }
}

//####################//
// LaneRule Functions //
//####################//

/** Adds a road mark entry to this lane.
*/
void
Lane::addLaneRuleEntry(LaneRule *ruleEntry)
{
	rules_.insert(ruleEntry->getSSectionStart(), ruleEntry);
	ruleEntry->setParentLane(this);
	addLaneChanges(Lane::CLN_LaneRulesChanged);
}

/** Returns the road mark entry at the given section coordinate (sSection).
*/
LaneRule *
Lane::getLaneRuleEntry(double sSection) const
{
	QMap<double, LaneRule *>::const_iterator i = rules_.upperBound(sSection);
	if (i == rules_.constBegin())
	{
		qDebug("WARNING: Trying to get lane road mark but coordinate is out of bounds!");
		return NULL;
	}
	else
	{
		--i;
		return i.value();
	}
}

/** Returns the end coordinate of the road mark
* containing the lane section coordinate sSection.
* In lane section coordinates [m].
* If the road mark is the last in the list, the end of
* the lane section will be returned. Otherwise the end of
* a road mark is the start coordinate of the successing one.
*/
double
Lane::getLaneRuleEnd(double sSection) const
{
	//sSection = 0.0; // ???
	QMap<double, LaneRule *>::const_iterator nextIt = rules_.upperBound(sSection);
	if (nextIt == rules_.constEnd())
	{
		return parentLaneSection_->getLength(); // lane section: [0.0, length]
	}
	else
	{
		return (*nextIt)->getSSectionStart();
	}
}

//####################//
// LaneAccess Functions //
//####################//

/** Adds a road mark entry to this lane.
*/
void
Lane::addLaneAccessEntry(LaneAccess *accessEntry)
{
	accesses_.insert(accessEntry->getSSectionStart(), accessEntry);
	accessEntry->setParentLane(this);
	addLaneChanges(Lane::CLN_LaneAccessChanged);
}

/** Returns the road mark entry at the given section coordinate (sSection).
*/
LaneAccess *
Lane::getLaneAccessEntry(double sSection) const
{
	QMap<double, LaneAccess *>::const_iterator i = accesses_.upperBound(sSection);
	if (i == accesses_.constBegin())
	{
		qDebug("WARNING: Trying to get lane road mark but coordinate is out of bounds!");
		return NULL;
	}
	else
	{
		--i;
		return i.value();
	}
}

/** Returns the end coordinate of the road mark
* containing the lane section coordinate sSection.
* In lane section coordinates [m].
* If the road mark is the last in the list, the end of
* the lane section will be returned. Otherwise the end of
* a road mark is the start coordinate of the successing one.
*/
double
Lane::getLaneAccessEnd(double sSection) const
{
	//sSection = 0.0; // ???
	QMap<double, LaneAccess *>::const_iterator nextIt = accesses_.upperBound(sSection);
	if (nextIt == accesses_.constEnd())
	{
		return parentLaneSection_->getLength(); // lane section: [0.0, length]
	}
	else
	{
		return (*nextIt)->getSSectionStart();
	}
}

//####################//
// Speed Functions    //
//####################//

/** Adds a speed entry to this lane.
*/
void
Lane::addSpeedEntry(LaneSpeed *speedEntry)
{
    speeds_.insert(speedEntry->getSOffset(), speedEntry);
    speedEntry->setParentLane(this);
    addLaneChanges(Lane::CLN_SpeedsChanged);
}

/** Returns the allowed velocity at the given section coordinate sSection.
*/
double
Lane::getSpeed(double sSection) const
{
    QMap<double, LaneSpeed *>::const_iterator i = speeds_.upperBound(sSection);
    if (i == speeds_.constBegin())
    {
        qDebug("WARNING: Trying to get lane speed but coordinate is out of bounds!");
        return -1.0;
    }
    else
    {
        --i;
        return i.value()->getMaxSpeed();
    }
}

/** Returns the end coordinate of the speed entry
* containing the lane section coordinate sSection.
* In lane section coordinates [m].
*/
double
Lane::getSpeedEnd(double sSection) const
{
    //sSection = 0.0; // ???
    QMap<double, LaneSpeed *>::const_iterator nextIt = speeds_.upperBound(sSection);
    if (nextIt == speeds_.constEnd())
    {
        return parentLaneSection_->getLength(); // lane section: [0.0, length]
    }
    else
    {
        return (*nextIt)->getSSectionStart();
    }
}

//###################//
// Height Functions  //
//###################//

/** Adds a lane height entry to this lane.
*/
void
Lane::addHeightEntry(LaneHeight *heightEntry)
{
    heights_.insert(heightEntry->getSOffset(), heightEntry);
    heightEntry->setParentLane(this);
    addLaneChanges(Lane::CLN_HeightsChanged);
}

/** Returns the end coordinate of the height entry
* containing the lane section coordinate sSection.
* In lane section coordinates [m].
*/
double
Lane::getHeightEnd(double sSection) const
{
    //sSection = 0.0; // ???
    QMap<double, LaneHeight *>::const_iterator nextIt = heights_.upperBound(sSection);
    if (nextIt == heights_.constEnd())
    {
        return parentLaneSection_->getLength(); // lane section: [0.0, length]
    }
    else
    {
        return (*nextIt)->getSSectionStart();
    }
}

//################//
// OBSERVER       //
//################//

/*! \brief Called after all Observers have been notified.
*
* Resets the change flags to 0x0.
*/
void
Lane::notificationDone()
{
    laneChanges_ = 0x0;
    DataElement::notificationDone();
}

/*! \brief Add one or more change flags.
*
*/
void
Lane::addLaneChanges(int changes)
{
    if (changes == CLN_WidthsChanged)
    {
        if (parentLaneSection_ && parentLaneSection_->getParentRoad())
            parentLaneSection_->getParentRoad()->addRoadChanges(RSystemElementRoad::CRD_ShapeChange);
    }
    if (changes)
    {
        laneChanges_ |= changes;
        notifyObservers();
    }
}

//###################//
// Prototype Pattern //
//###################//

/*! \brief Creates and returns a deep copy clone of this object.
*
*/
Lane *
Lane::getClone() const
{
    // Lane //
    //
    Lane *clone = new Lane(id_, type_, level_, predecessorId_, successorId_);

    // LanesWidth //
    //
    foreach (LaneWidth *child, widths_)
    {
        clone->addWidthEntry(child->getClone());
    }

    // LanesRoadMark //
    //
    foreach (LaneRoadMark *child, marks_)
    {
        clone->addRoadMarkEntry(child->getClone());
    }

    // LanesSpeed //
    //
    foreach (LaneSpeed *child, speeds_)
    {
        clone->addSpeedEntry(child->getClone());
    }

    return clone;
}

/*! \brief Creates and returns a deep copy clone of this object.
*
*/
Lane *
Lane::getClone(double sOffsetStart, double sOffsetEnd) const
{
    // Lane //
    //
    Lane *clone = new Lane(id_, type_, level_, predecessorId_, successorId_);

    // LanesWidth //
    //
    foreach (LaneWidth *child, widths_)
    {
        double parentStart = child->getParentLane()->getParentLaneSection()->getSStart();
        double childStart = parentStart + child->getSOffset();
        if (childStart < sOffsetStart)
        {
            double childEnd = child->getParentLane()->getParentLaneSection()->getSEnd() + child->getSOffset();
            // a) Starts and ends before cloned region //
            //
            if (childEnd < sOffsetStart)
            {
                continue; // do not add
            }

            // b) Starts before cloned region, ends in it //
            //
            else
            {
                LaneWidth *newEntry = new LaneWidth(0.0, child->getWidth(sOffsetStart - parentStart), child->getSlope(sOffsetStart - parentStart), child->getCurvature(sOffsetStart - parentStart) / 2.0, child->getD());
                newEntry->setSOffset(0.0);
                clone->addWidthEntry(newEntry);
            }
        }
        else if (childStart < sOffsetEnd)
        {
            // c) Starts in cloned region //
            //
            double newSStart = childStart - sOffsetStart;
            if (fabs(newSStart) < NUMERICAL_ZERO6)
            {
                newSStart = 0.0; // round to zero, so it isn't negative (e.g. -1e-7)
            }
            LaneWidth *newEntry = child->getClone();
            newEntry->setSOffset(newSStart);
            clone->addWidthEntry(newEntry);
        }
        else
        {
            // d) Starts behind cloned region //
            //
            continue; // do not add
        }
    }

    // LanesRoadMark //
    //
    foreach (LaneRoadMark *child, marks_)
    {
        double childStart = child->getParentLane()->getParentLaneSection()->getSStart() + child->getSOffset();
        if (childStart < sOffsetStart)
        {

            double childEnd = child->getParentLane()->getParentLaneSection()->getSEnd() + child->getSOffset();

            // a) Starts and ends before cloned region //)
            //
            if (childEnd < sOffsetStart)
            {
                continue; // do not add
            }

            // b) Starts before cloned region, ends in it //
            //
            else
            {
                LaneRoadMark *newEntry = child->getClone();
                newEntry->setSOffset(0.0);
                clone->addRoadMarkEntry(newEntry);
            }
        }
        else if (childStart < sOffsetEnd)
        {
            // c) Starts in cloned region //
            //
            double newSStart = childStart - sOffsetStart;
            if (fabs(newSStart) < NUMERICAL_ZERO6)
            {
                newSStart = 0.0; // round to zero, so it isn't negative (e.g. -1e-7)
            }
            LaneRoadMark *newEntry = child->getClone();
            newEntry->setSOffset(newSStart);
            clone->addRoadMarkEntry(newEntry);
        }
        else
        {
            // d) Starts behind cloned region //
            //
            continue; // do not add
        }
    }

    // LanesSpeed //
    //
    foreach (LaneSpeed *child, speeds_)
    {
        if (child->getSSectionStart() < sOffsetStart)
        {
            // a) Starts and ends before cloned region //
            //
            if (child->getSSectionEnd() < sOffsetStart)
            {
                continue; // do not add
            }

            // b) Starts before cloned region, ends in it //
            //
            else
            {
                LaneSpeed *newEntry = child->getClone();
                newEntry->setSOffset(0.0);
                clone->addSpeedEntry(newEntry);
            }
        }
        else if (child->getSSectionStart() < sOffsetEnd)
        {
            // c) Starts in cloned region //
            //
            double newSStart = child->getSOffset() - sOffsetStart;
            if (fabs(newSStart) < NUMERICAL_ZERO6)
            {
                newSStart = 0.0; // round to zero, so it isn't negative (e.g. -1e-7)
            }
            LaneSpeed *newEntry = child->getClone();
            newEntry->setSOffset(newSStart);
            clone->addSpeedEntry(newEntry);
        }
        else
        {
            // d) Starts behind cloned region //
            //
            continue; // do not add
        }
    }

    return clone;
}

//###################//
// Visitor Pattern   //
//###################//

/*!
* Accepts a visitor and passes it to all child
* nodes if autoTraverse is true.
*/
void
Lane::accept(Visitor *visitor)
{
    visitor->visit(this);
}

/*!
* Accepts a visitor and passes it to all child nodes.
*/
void
Lane::acceptForChildNodes(Visitor *visitor)
{
    acceptForWidths(visitor);
    acceptForRoadMarks(visitor);
    acceptForSpeeds(visitor);
    acceptForHeights(visitor);
	acceptForRules(visitor);
	acceptForAccess(visitor);
}

/*! Accepts a visitor and passes it to the width entries.
*/
void
Lane::acceptForWidths(Visitor *visitor)
{
    foreach (LaneWidth *child, widths_)
        child->accept(visitor);
}

/*! Accepts a visitor and passes it to the road mark entries.
*/
void
Lane::acceptForRoadMarks(Visitor *visitor)
{
    foreach (LaneRoadMark *child, marks_)
        child->accept(visitor);
}

/*! Accepts a visitor and passes it to the speed entries.
*/
void
Lane::acceptForSpeeds(Visitor *visitor)
{
    foreach (LaneSpeed *child, speeds_)
        child->accept(visitor);
}

/*! Accepts a visitor and passes it to the height entries.
*/
void
Lane::acceptForHeights(Visitor *visitor)
{
    foreach (LaneHeight *child, heights_)
        child->accept(visitor);
}

/*! Accepts a visitor and passes it to the road mark entries.
*/
void
Lane::acceptForRules(Visitor *visitor)
{
	foreach(LaneRule *child, rules_)
		child->accept(visitor);
}

/*! Accepts a visitor and passes it to the road mark entries.
*/
void
Lane::acceptForAccess(Visitor *visitor)
{
	foreach(LaneAccess *child, accesses_)
		child->accept(visitor);
}
