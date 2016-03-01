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

#include "lanesection.hpp"

#include "src/data/roadsystem/rsystemelementroad.hpp"

#include "lane.hpp"

//####################//
// Constructors       //
//####################//

/*!
 *
 */
LaneSection::LaneSection(double s)
    : RoadSection(s),
      laneSectionChanges_(0x0)
{
}

LaneSection::LaneSection(double s, const LaneSection *oldLaneSection) // create new LaneSection at pos s of oldLaneSection
    : RoadSection(s),
      laneSectionChanges_(0x0)
{
    setParentRoad(oldLaneSection->getParentRoad());
    foreach (Lane *child, oldLaneSection->lanes_)
    {
        addLane(child->getClone(s, getSEnd()));
    }
}

LaneSection::LaneSection(double s, const LaneSection *laneSectionLow,
        const LaneSection *laneSectionHigh) // create new LaneSection as e merger between low and high
    : RoadSection(s),
      laneSectionChanges_(0x0)
{
    setParentRoad(laneSectionLow->getParentRoad());
    foreach (Lane *childLow, laneSectionLow->lanes_)
    {
        Lane *childHigh = laneSectionHigh->lanes_.value(childLow->getId(), NULL);

        Lane *newLane = childLow->getClone(s, laneSectionHigh->getSEnd());
        // TODO set width
        addLane(newLane);
    }
}

LaneSection::~LaneSection()
{
    foreach (Lane *child, lanes_)
    {
        delete child;
    }
}

//####################//
// Section Functions  //
//####################//

/*! \brief Returns the end coordinate of this section.
 * In road coordinates [m].
 *
 */
double LaneSection::getSEnd() const
{
    return getParentRoad()->getLaneSectionEnd(getSStart());
}

/*! \brief Returns the length coordinate of this section.
 * In [m].
 *
 */
double LaneSection::getLength() const
{
    return getParentRoad()->getLaneSectionEnd(getSStart()) - getSStart();
}

/*! \brief Returns first lane
 *
 */
Lane *
LaneSection::getFirst() const
{
    if (lanes_.isEmpty())
    {
        return NULL;
    }

    QMap<int, Lane *>::const_iterator it = lanes_.constBegin();
    return it.value();
}

/*! \brief Returns last lane
 *
 */
Lane *
LaneSection::getLast() const
{
    if (lanes_.isEmpty())
    {
        return NULL;
    }

    QMap<int, Lane *>::const_iterator it = lanes_.constEnd();
    it--;
    return it.value();
}

/*! \brief Returns the adjacent lane with id greater than the argument given
 *
 */
Lane *
LaneSection::getNextUpper(int id) const
{
    QMap<int, Lane *>::const_iterator i = lanes_.upperBound(id);
    if (i == lanes_.constEnd())
    {
        return NULL;
    }

    return i.value();
}

/*! \brief Returns the adjacent lane with id less than the argument given
 *
 */
Lane *
LaneSection::getNextLower(int id) const
{
    QMap<int, Lane *>::const_iterator i = lanes_.lowerBound(id);
    if (i == lanes_.constBegin())
    {
        return NULL;
    }
    i--;

    return i.value();
}

//####################//
// Lane Functions     //
//####################//

Lane *
LaneSection::getLane(int id) const
{
    return lanes_.value(id, NULL);
}

int LaneSection::getLaneId(double s, double t)
{
    int i = 0;
    double width = 0;
    Lane * lane = getLane(i);
    if (t < 0.0)
    {
        while (width < fabs(t))
        {
            lane = getNextLower(i);
            if (!lane)
            {
                return i;
            }
            width += getLaneWidth(--i, s);
        }
    }
    else
    {
        while (width < t)
        {
            lane = getNextUpper(i);
            if (!lane)
            {
                return i;
            }
            width += getLaneWidth(++i, s);
        }
    }
    return i;
}

void LaneSection::setLanes(QMap<int, Lane *> newLanes)
{
    foreach (Lane *child, lanes_)
    {
        child->setParentLaneSection(NULL);
    }

    foreach (Lane *child, newLanes)
    {
        child->setParentLaneSection(this);
    }

    lanes_ = newLanes;
    addLaneSectionChanges(LaneSection::CLS_LanesChanged);
}

/*! \brief Adds a lane to this lane section.
 *
 */
void LaneSection::addLane(Lane *lane)
{
    if (lanes_.contains(lane->getId()))
    {
        qDebug(
                "ERROR 1010151532! A lane with the same ID already exists! Road::%s", getParentRoad()->getName().toUtf8().constData());
        return;
    }
    lanes_.insert(lane->getId(), lane);
    lane->setParentLaneSection(this);
    addLaneSectionChanges(LaneSection::CLS_LanesChanged);
}

/*! \brief Adds a lane to this lane section.
 *
 */
void LaneSection::removeLane(Lane *lane)
{
    if (!lanes_.contains(lane->getId()))
    {
        qDebug("ERROR 1010151532! A lane this ID does not exist!");
        return;
    }
    lanes_.remove(lane->getId());
    lane->setParentLaneSection(NULL);
    addLaneSectionChanges(LaneSection::CLS_LanesChanged);
}

/*! \brief Returns the width of the lane with the ID lane at the given road coordinate s.
 *
 * \param lane	ID of the lane.
 * \param s		Road coordinate for which the width should be calculated [m].
 *
 * \returns If the lane ID is 0 (i.e. the center lane) the function returns 0.0 by definition.
 * If there is no lane with the given ID, 0.0 is returned.
 */
double LaneSection::getLaneWidth(int lane, double s) const
{
    // width for lane 0 is 0.0 by default
    if (lane == 0)
    {
        return 0.0;
    }

    // check for lane and (if found) return width
    Lane *laneEntry = lanes_.value(lane, NULL);
    if (laneEntry)
    {
        s = s - getSStart();
        if (s < 0.0)
            s = 0.0; // clamp
        return laneEntry->getWidth(s);
    }
    else
    {
        return 0.0;
    }
}

double LaneSection::getLaneSpanWidth(int fromLane, int toLane, double s) const
{
    if (fromLane > toLane)
    {
        int tmp = fromLane;
        fromLane = toLane;
        toLane = tmp;
    }

    double width = 0.0;
    for (int i = fromLane; i <= toLane; ++i)
    {
        width += getLaneWidth(i, s);
    }
    return width;
}

void LaneSection::checkAndFixLanes()
{
// Lane ids should be subsequent
//
    for (int i = 1; i < lanes_.size(); i++)
    {
        if (!lanes_.contains(i)) // not found
        {
            for (int n = 1; n < lanes_.size() - i; n++)
            {
                if (lanes_.contains(i + n))
                {
                    Lane *lane = lanes_.value(i + n, NULL);
                    if (lane)
                    {
                        removeLane(lane);
                        lane->setId(i);
                        addLane(lane);
                    }
                }
            }
        }
    }
    for (int i = -1; i > -(lanes_.size()); i--)
    {
        if (!lanes_.contains(i)) // not found
        {
            for (int n = 1; n < lanes_.size() - i; n++)
            {
                if (lanes_.contains(i - n))
                {
                    Lane *lane = lanes_.value(i - n, NULL);
                    if (lane)
                    {
                        removeLane(lane);
                        lane->setId(i);
                        addLane(lane);
                    }
                }
            }
        }
    }
}

// Returns distance from road center to mid of lane
//
double LaneSection::getTValue(Lane * lane, double s, double laneWidth)
{
    double t = 0.0;

    if (lane->getId() < 0)
    {
        if (laneWidth < NUMERICAL_ZERO3)
        {
            if (lane->getId() == getRightmostLaneId())
            {
                t = NUMERICAL_ZERO3;
            }
            else
            {
                t = -NUMERICAL_ZERO3;
            }
        }
        else
        {
            t = -laneWidth / 2;
        }

        t = -getLaneSpanWidth(lane->getId() + 1, 0, s) + t;
    }
    else if (lane->getId() > 0)
    {
        if (laneWidth < NUMERICAL_ZERO3)
        {
            if (lane->getId() == getLeftmostLaneId())
            {
                t = -NUMERICAL_ZERO3;
            }
            else
            {
                t = NUMERICAL_ZERO3;
            }
        }
        else
        {
            t = laneWidth / 2;
        }

        t = getLaneSpanWidth(0, lane->getId() - 1, s) + t;
    }

    return t;
}

int LaneSection::getRightmostLaneId() const
{
    QMap<int, Lane *>::const_iterator laneIt = lanes_.begin();
    if (laneIt != lanes_.end())
    {
        return (*laneIt)->getId();
    }
    else
    {
        return Lane::NOLANE;
    }
}

int LaneSection::getLeftmostLaneId() const
{
    QMap<int, Lane *>::const_iterator laneIt = lanes_.end();
    if (laneIt != lanes_.begin())
    {
        return (*(--laneIt))->getId();
    }
    else
    {
        return Lane::NOLANE;
    }
}

//################//
// OBSERVER       //
//################//

/*! \brief Called after all Observers have been notified.
 *
 * Resets the change flags to 0x0.
 */
void LaneSection::notificationDone()
{
    laneSectionChanges_ = 0x0;
    RoadSection::notificationDone();
}

/*! \brief Add one or more change flags.
 *
 */
void LaneSection::addLaneSectionChanges(int changes)
{
    if (changes)
    {
        laneSectionChanges_ |= changes;
        notifyObservers();
    }
}

//###################//
// Prototype Pattern //
//###################//

/*! \brief Creates and returns a deep copy clone of this object.
 *
 */
LaneSection *
LaneSection::getClone() const
{
// LaneSection //
//
    LaneSection *clone = new LaneSection(getSStart());

// Lanes //
//
    foreach (Lane *child, lanes_){
    clone->addLane(child->getClone());
}

    return clone;
}

/*! \brief Creates and returns a deep copy clone of this object.
 *
 */
LaneSection *
LaneSection::getClone(double sStart, double sEnd) const
{
// LaneSection //
//
    LaneSection *clone = new LaneSection(getSStart());

// Lanes //
//
    foreach (Lane *child, lanes_)
    {
        clone->addLane(child->getClone(sStart, sEnd));
    }

    return clone;
}

//###################//
// Visitor Pattern   //
//###################//

/*! \brief Accepts a visitor for this lane section.
 *
 * \param visitor The visitor that will be visited.
 */
void LaneSection::accept(Visitor *visitor)
{
    visitor->visit(this);
}

/*! \brief Accepts a visitor for the lanes.
 */
void LaneSection::acceptForLanes(Visitor *visitor)
{
    foreach (Lane *child, lanes_)
    {
        child->accept(visitor);
    }
}
