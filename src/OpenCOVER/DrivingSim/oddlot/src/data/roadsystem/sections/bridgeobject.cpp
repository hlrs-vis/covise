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

#include "bridgeobject.hpp"

#include "src/data/roadsystem/rsystemelementroad.hpp"

//####################//
// Constructors       //
//####################//

Bridge::Bridge(const QString &id, const QString &file, const QString &name, int type, double s, double length)
    : RoadSection(s)
    , id_(id)
    , fileName_(file)
    , name_(name)
    , type_(type)
    , length_(length)
{
}

/*!
* Returns the end coordinate of this section.
* In road coordinates [m].
*
*/
double
Bridge::getSEnd() const
{

    return getSStart() + length_;
}

QString
Bridge::getNewId(const QString &name)
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
Bridge::notificationDone()
{
    bridgeChanges_ = 0x0;
    RoadSection::notificationDone();
}

/*! \brief Add one or more change flags.
*
*/
void
Bridge::addBridgeChanges(int changes)
{
    if (changes)
    {
        bridgeChanges_ |= changes;
        notifyObservers();
    }
}

//###################//
// Prototype Pattern //
//###################//

/*! \brief Creates and returns a deep copy clone of this object.
*
*/
Bridge *
Bridge::getClone()
{
    // Bridge //
    //
    Bridge *clone = new Bridge(id_, fileName_, name_, type_, getSStart(), length_);

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
Bridge::accept(Visitor *visitor)
{
    visitor->visit(this);
}
