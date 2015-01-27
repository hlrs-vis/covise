/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   02.06.2010
**
**************************************************************************/

#include "poolvehicle.hpp"

PoolVehicle::PoolVehicle(
    const QString &id,
    double numerator)
    : DataElement()
    , roadVehicleChanges_(0x0)
    , id_(id)
    , numerator_(numerator)
{
}

PoolVehicle::~PoolVehicle()
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
PoolVehicle::notificationDone()
{
    roadVehicleChanges_ = 0x0;
    DataElement::notificationDone();
}

/*! \brief Add one or more change flags.
*
*/
void
PoolVehicle::addPoolVehicleChanges(int changes)
{
    if (changes)
    {
        roadVehicleChanges_ |= changes;
        notifyObservers();
    }
}

//##################//
// Visitor Pattern  //
//##################//

/*! \brief Accepts a visitor.
*
* With autotraverse: visitor will be send to roads, fiddleyards, etc.
* Without: accepts visitor as 'this'.
*/
void
PoolVehicle::accept(Visitor *visitor)
{
    visitor->visit(this);
}
