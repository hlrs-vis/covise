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

#include "signalobject.hpp"

#include "src/data/roadsystem/rsystemelementroad.hpp"

//####################//
// Constructors       //
//####################//

Signal::Signal(const QString &id, const QString &name, double s, double t, bool dynamic, OrientationType orientation, double zOffset, QString country, int type, const QString &typeSubclass, int subtype, double value, bool pole, int size, int validFromLane, int validToLane, double probability, double resetTime)
    : RoadSection(s)
    , id_(id)
    , name_(name)
{
    signalProps_.t = t;
    signalProps_.dynamic = dynamic;
    signalProps_.orientation = orientation;
    signalProps_.zOffset = zOffset;
    signalProps_.country = country;
    signalProps_.type = type;
    signalProps_.typeSubclass = typeSubclass;
    signalProps_.subtype = subtype;
    signalProps_.value = value;
    signalProps_.pole = pole;
    signalProps_.size = size;

    validity_.fromLane = validFromLane;
    validity_.toLane = validToLane;

    signalUserData_.crossProb = probability;
    signalUserData_.resetTime = resetTime;
}

Signal::Signal(const QString &id, const QString &name, double s, SignalProperties &signalProps, Validity &validity, SignalUserData &userData)
    : RoadSection(s)
    , id_(id)
    , name_(name)
    , signalProps_(signalProps)
    , validity_(validity)
    , signalUserData_(userData)
{
}

QString
Signal::getNewId(const QString &name)
{
    QStringList parts = id_.split("_");
    QString newId = parts.at(0) + "_" + parts.at(1) + "_" + name;

    return newId;
}

//##################//
// Observer Pattern //
//##################//

/*! \brief Called after all Observers have been notified.
*
* Resets the change flags to 0x0.
*/
void
Signal::notificationDone()
{
    signalChanges_ = 0x0;
    RoadSection::notificationDone();
}

/*! \brief Add one or more change flags.
*
*/
void
Signal::addSignalChanges(int changes)
{
    if (changes)
    {
        signalChanges_ |= changes;
        notifyObservers();
    }
}

//###################//
// Prototype Pattern //
//###################//

/*! \brief Creates and returns a deep copy clone of this object.
*
*/
Signal *
Signal::getClone()
{
    // Signal //
    //
    Signal *clone = new Signal(id_, name_, getSStart(), signalProps_, validity_, signalUserData_);

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
Signal::accept(Visitor *visitor)
{
    visitor->visit(this);
}
