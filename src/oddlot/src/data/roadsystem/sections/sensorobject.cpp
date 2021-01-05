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

#include "sensorobject.hpp"

#include "src/data/roadsystem/rsystemelementroad.hpp"

//####################//
// Constructors       //
//####################//

Sensor::Sensor(const QString &id, double s)
    : RoadSection(s)
    , id_(id)
{
}

//##################//
// Observer Pattern //
//##################//

/*! \brief Called after all Observers have been notified.
*
* Resets the change flags to 0x0.
*/
void
Sensor::notificationDone()
{
    sensorChanges_ = 0x0;
    RoadSection::notificationDone();
}

/*! \brief Add one or more change flags.
*
*/
void
Sensor::addSensorChanges(int changes)
{
    if (changes)
    {
        sensorChanges_ |= changes;
        notifyObservers();
    }
}

//###################//
// Prototype Pattern //
//###################//

/*! \brief Creates and returns a deep copy clone of this object.
*
*/
Sensor *
Sensor::getClone()
{
    // Sensor //
    //
    Sensor *clone = new Sensor(id_, s_);

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
Sensor::accept(Visitor *visitor)
{
    visitor->visit(this);
}
