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

#include "carpool.hpp"

#include "pool.hpp"
#include "vehiclesystem.hpp"

CarPool::CarPool()
    : DataElement()
    , parentVehicleSystem_(NULL)
{
}

CarPool::~CarPool()
{
}

void
CarPool::addPool(Pool *pool)
{
    addCarPoolChanges(CarPool::CVR_PoolChanged);
    pools_.insert(parentVehicleSystem_->getUniqueId(pool->getID()), pool);

    pool->setParentCarPool(this);
}

/*! \brief Accepts a visitor and passes it to child nodes.
*/
void
CarPool::acceptForChildNodes(Visitor *visitor)
{
    foreach (Pool *child, pools_)
    {
        child->accept(visitor);
    }
}

//##################//
// ProjectData      //
//##################//

void
CarPool::setParentVehicleSystem(VehicleSystem *vehicleSystem)
{
    parentVehicleSystem_ = vehicleSystem;
    setParentElement(vehicleSystem);
    addCarPoolChanges(CarPool::CVR_VehicleSystemChanged);
}

//##################//
// Observer Pattern //
//##################//

/*! \brief Called after all Observers have been notified.
*
* Resets the change flags to 0x0.
*/
void
CarPool::notificationDone()
{
    carPoolChanges_ = 0x0;
    DataElement::notificationDone();
}

/*! \brief Add one or more change flags.
*
*/
void
CarPool::addCarPoolChanges(int changes)
{
    if (changes)
    {
        carPoolChanges_ |= changes;
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
CarPool::accept(Visitor *visitor)
{
    visitor->visit(this);
}
