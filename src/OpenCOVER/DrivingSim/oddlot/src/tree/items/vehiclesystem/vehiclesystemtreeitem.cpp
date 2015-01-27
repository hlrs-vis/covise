/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   11/17/2010
**
**************************************************************************/

#include "vehiclesystemtreeitem.hpp"

// Data //
//
#include "src/data/vehiclesystem/vehiclesystem.hpp"
#include "src/data/vehiclesystem/vehiclegroup.hpp"

// Tree //
//
#include "vehiclegrouptreeitem.hpp"

VehicleSystemTreeItem::VehicleSystemTreeItem(ProjectTree *projectTree, VehicleSystem *vehicleSystem, QTreeWidgetItem *rootItem)
    : ProjectTreeItem(NULL, vehicleSystem, rootItem)
    , // no direct parent, rootItem is foster parent
    projectTree_(projectTree)
    , vehicleSystem_(vehicleSystem)
    , rootItem_(rootItem)
{
    init();
}

VehicleSystemTreeItem::~VehicleSystemTreeItem()
{
}

void
VehicleSystemTreeItem::init()
{
    // Text //
    //
    setText(0, tr("VehicleSystem"));

    // Maps //
    //
    vehicleGroupsItem_ = new QTreeWidgetItem(this);
    vehicleGroupsItem_->setText(0, tr("Vehicle Groups"));

    foreach (VehicleGroup *group, vehicleSystem_->getVehicleGroups())
    {
        new VehicleGroupTreeItem(this, group, vehicleGroupsItem_);
    }
}

//##################//
// Observer Pattern //
//##################//

void
VehicleSystemTreeItem::updateObserver()
{
    // Parent //
    //
    ProjectTreeItem::updateObserver();

    if (isInGarbage())
    {
        return; // will be deleted anyway
    }

    // VehicleSystem //
    //
    int changes = getVehicleSystem()->getVehicleSystemChanges();

    if (changes & VehicleSystem::CVS_VehicleGroupsChanged)
    {
        // A VehicleGroup has been added (or deleted - but that will be handled by the item itself).
        //
        foreach (VehicleGroup *group, vehicleSystem_->getVehicleGroups())
        {
            if ((group->getDataElementChanges() & DataElement::CDE_DataElementCreated)
                || (group->getDataElementChanges() & DataElement::CDE_DataElementAdded))
            {
                new VehicleGroupTreeItem(this, group, vehicleGroupsItem_);
            }
        }
    }
}
