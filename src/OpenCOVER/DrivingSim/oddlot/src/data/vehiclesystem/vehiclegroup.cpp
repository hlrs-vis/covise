/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   07.06.2010
**
**************************************************************************/

#include "vehiclegroup.hpp"

// Data //
//
#include "vehiclesystem.hpp"
#include "roadvehicle.hpp"

VehicleGroup::VehicleGroup(double rangeLOD)
    : DataElement()
    , parentVehicleSystem_(NULL)
    , vehicleGroupChanges_(0x0)
    , rangeLOD_(rangeLOD)
    , passThresholdSet_(false)
{
}

VehicleGroup::~VehicleGroup()
{
    // Delete child nodes //
    //
    foreach (RoadVehicle *child, roadVehicles_)
    {
        delete child;
    }
}

//##################//
// VehicleGroup     //
//##################//

void
VehicleGroup::addRoadVehicle(RoadVehicle *roadVehicle)
{
    addVehicleGroupChanges(VehicleGroup::CVG_RoadVehicleChanged);
    roadVehicles_.insert(parentVehicleSystem_->getUniqueId(roadVehicle->getId()), roadVehicle);

    roadVehicle->setParentVehicleGroup(this);
}

//##################//
// VehicleSystem    //
//##################//

void
VehicleGroup::setParentVehicleSystem(VehicleSystem *vehicleSystem)
{
    parentVehicleSystem_ = vehicleSystem;
    setParentElement(vehicleSystem);
    addVehicleGroupChanges(VehicleGroup::CVG_VehicleSystemChanged);
}

//##################//
// Observer Pattern //
//##################//

/*! \brief Called after all Observers have been notified.
*
* Resets the change flags to 0x0.
*/
void
VehicleGroup::notificationDone()
{
    vehicleGroupChanges_ = 0x0;
    DataElement::notificationDone();
}

/*! \brief Add one or more change flags.
*
*/
void
VehicleGroup::addVehicleGroupChanges(int changes)
{
    if (changes)
    {
        vehicleGroupChanges_ |= changes;
        notifyObservers();
    }
}

//##################//
// Visitor Pattern  //
//##################//

/*! \brief Accepts a visitor.
*
*/
void
VehicleGroup::accept(Visitor *visitor)
{
    visitor->visit(this);
}

/*! \brief Accepts a visitor and passes it to child nodes.
*/
void
VehicleGroup::acceptForChildNodes(Visitor *visitor)
{
    acceptForRoadVehicles(visitor);
}

/*! \brief Accepts a visitor and passes it to child nodes.
*/
void
VehicleGroup::acceptForRoadVehicles(Visitor *visitor)
{
    foreach (RoadVehicle *child, roadVehicles_)
    {
        child->accept(visitor);
    }
}

//##################//
// Static Functions //
//##################//

double VehicleGroup::defaultRangeLOD = 500000.0;
