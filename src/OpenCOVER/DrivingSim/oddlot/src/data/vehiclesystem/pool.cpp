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

#include "pool.hpp"

#include "poolvehicle.hpp"
#include "carpool.hpp"

Pool::Pool(
    const QString &name,
    const QString &id,
    double velocity,
    double velocityDeviance,
    double numerator)
    : DataElement()
    , parentCarPool_(NULL)
    , poolChanges_(0)
    , name_(name)
    , id_(id)
    , velocity_(velocity)
    , velocityDeviance_(velocityDeviance)
    , numerator_(numerator)
{
}

Pool::~Pool()
{
}

void Pool::addVehicle(PoolVehicle *poolVehicle)
{
    vehicles_.append(poolVehicle);

    addPoolChanges(Pool::CVR_VehiclesChanged);
}

//##################//
// ProjectData      //
//##################//

void
Pool::setParentCarPool(CarPool *carPool)
{
    parentCarPool_ = carPool;
    setParentElement(carPool);
    addPoolChanges(Pool::CVR_CarPoolChanged);
}

//##################//
// Observer Pattern //
//##################//

/*! \brief Called after all Observers have been notified.
*
* Resets the change flags to 0x0.
*/
void
Pool::notificationDone()
{
    poolChanges_ = 0x0;
    DataElement::notificationDone();
}

/*! \brief Add one or more change flags.
*
*/
void
Pool::addPoolChanges(int changes)
{
    if (changes)
    {
        poolChanges_ |= changes;
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
Pool::accept(Visitor *visitor)
{
    visitor->visit(this);
}
