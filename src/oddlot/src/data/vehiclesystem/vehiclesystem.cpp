/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   01.06.2010
**
**************************************************************************/

#include "vehiclesystem.hpp"

// Data //
//
#include "src/data/projectdata.hpp"
#include "vehiclegroup.hpp"
#include "carpool.hpp"

VehicleSystem::VehicleSystem()
    : DataElement()
    , vehicleSystemChanges_(0x0)
    , parentProjectData_(NULL)
    , idCount_(0)
{
}

VehicleSystem::~VehicleSystem()
{
    // Delete child nodes //
    //
    foreach (VehicleGroup *child, vehicleGroups_)
    {
        delete child;
    }
}

//##################//
// VehicleGroups    //
//##################//

void
VehicleSystem::addVehicleGroup(VehicleGroup *vehicleGroup)
{
    vehicleGroups_.append(vehicleGroup);
    addVehicleSystemChanges(VehicleSystem::CVS_VehicleGroupsChanged);

    vehicleGroup->setParentVehicleSystem(this);
}

void VehicleSystem::setCarPool(CarPool *carPool)
{
    carPool_ = carPool;
    addVehicleSystemChanges(VehicleSystem::CVS_CarPoolChanged);

    carPool->setParentVehicleSystem(this);
}

//##################//
// IDs              //
//##################//

const QString
VehicleSystem::getUniqueId(const QString &suggestion)
{
    // Try suggestion //
    //
    if (!suggestion.isNull())
    {
        if (!ids_.contains(suggestion))
        {
            ids_.append(suggestion);
            return suggestion;
        }
    }

    // Create new one //
    //
    QString id = QString("veh%1").arg(idCount_);
    while (ids_.contains(id))
    {
        id = QString("veh%1").arg(idCount_);
        ++idCount_;
    }
    ++idCount_;
    ids_.append(id);
    return id;
}

//##################//
// ProjectData      //
//##################//

void
VehicleSystem::setParentProjectData(ProjectData *projectData)
{
    parentProjectData_ = projectData;
    setParentElement(projectData);
    addVehicleSystemChanges(VehicleSystem::CVS_ProjectDataChanged);
}

//##################//
// Observer Pattern //
//##################//

/*! \brief Called after all Observers have been notified.
*
* Resets the change flags to 0x0.
*/
void
VehicleSystem::notificationDone()
{
    vehicleSystemChanges_ = 0x0;
    DataElement::notificationDone();
}

/*! \brief Add one or more change flags.
*
*/
void
VehicleSystem::addVehicleSystemChanges(int changes)
{
    if (changes)
    {
        vehicleSystemChanges_ |= changes;
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
VehicleSystem::accept(Visitor *visitor)
{
    visitor->visit(this);
}

/*! \brief Accepts a visitor and passes it to child nodes.
*/
void
VehicleSystem::acceptForChildNodes(Visitor *visitor)
{
    acceptForVehicleGroups(visitor);
}

/*! \brief Accepts a visitor and passes it to child nodes.
*/
void
VehicleSystem::acceptForVehicleGroups(Visitor *visitor)
{
    foreach (VehicleGroup *child, vehicleGroups_)
    {
        child->accept(visitor);
    }
}
