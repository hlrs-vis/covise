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

#include "crosswalkobject.hpp"

#include "src/data/roadsystem/rsystemelementroad.hpp"

//####################//
// Constructors       //
//####################//

Crosswalk::Crosswalk(const odrID &id, const QString &name, double s, double length)
    : RoadSection(s)
    , id_(id)
    , name_(name)
    , s_(s)
    , length_(length)
    , crossProbSet_(false)
    , resetTimeSet_(false)
    , typeSet_(false)
    , debugLvlSet_(false)
    , fromSet_(false)
    , toSet_(false)
{
}

/*!
* Returns the end coordinate of this section.
* In road coordinates [m].
*
*/
double
Crosswalk::getSEnd() const
{
    return s_ + length_;
}

//##################//
// Observer Pattern //
//##################//

/*! \brief Called after all Observers have been notified.
*
* Resets the change flags to 0x0.
*/
void
Crosswalk::notificationDone()
{
    crosswalkChanges_ = 0x0;
    RoadSection::notificationDone();
}

/*! \brief Add one or more change flags.
*
*/
void
Crosswalk::addCrosswalkChanges(int changes)
{
    if (changes)
    {
        crosswalkChanges_ |= changes;
        notifyObservers();
    }
}

//###################//
// Prototype Pattern //
//###################//

/*! \brief Creates and returns a deep copy clone of this object.
*
*/
Crosswalk *
Crosswalk::getClone()
{
    // Crosswalk //
    //
    Crosswalk *clone = new Crosswalk(id_, name_, s_, length_);
    if (crossProbSet_)
        clone->setCrossProb(crossProb_);
    if (resetTimeSet_)
        clone->setResetTime(resetTime_);
    if (typeSet_)
        clone->setType(type_);
    if (debugLvlSet_)
        clone->setDebugLvl(debugLvl_);
    if (fromSet_)
        clone->setFromLane(fromLane_);
    if (toSet_)
        clone->setToLane(toLane_);

    return clone;
}

//###################//
// Visitor Pattern   //
//###################//

/*!
* Accepts a visitor for this section.
*
* \param visitor The visitor that will be visited.
*/
void
Crosswalk::accept(Visitor *visitor)
{
    visitor->visit(this);
}
