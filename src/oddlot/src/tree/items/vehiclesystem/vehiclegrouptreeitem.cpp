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

#include "vehiclegrouptreeitem.hpp"

// Data //
//
#include "src/data/vehiclesystem/vehiclegroup.hpp"
#include "src/data/vehiclesystem/roadvehicle.hpp"

// Tree //
//
#include "vehiclesystemtreeitem.hpp"
#include "roadvehicletreeitem.hpp"

VehicleGroupTreeItem::VehicleGroupTreeItem(VehicleSystemTreeItem *parent, VehicleGroup *group, QTreeWidgetItem *fosterParent)
    : ProjectTreeItem(parent, group, fosterParent)
    , vehicleSystemTreeItem_(parent)
    , group_(group)
{
    init();
}

VehicleGroupTreeItem::~VehicleGroupTreeItem()
{
}

void
VehicleGroupTreeItem::init()
{
    // Text //
    //
    updateText();

    // RoadVehicles //
    //
    foreach (RoadVehicle *child, group_->getRoadVehicles())
    {
        new RoadVehicleTreeItem(this, child, NULL);
    }
}

void
VehicleGroupTreeItem::updateText()
{
    setText(0, tr("VehicleGroup"));
}

//##################//
// Observer Pattern //
//##################//

void
VehicleGroupTreeItem::updateObserver()
{
    // Parent //
    //
    ProjectTreeItem::updateObserver();
    if (isInGarbage())
    {
        return; // will be deleted anyway
    }

    // Get change flags //
    //
    int changes = group_->getVehicleGroupChanges();
    if ((changes & VehicleGroup::CVG_RoadVehicleChanged))
    {
        foreach (RoadVehicle *child, group_->getRoadVehicles())
        {
            if ((child->getDataElementChanges() & DataElement::CDE_DataElementCreated)
                || (child->getDataElementChanges() & DataElement::CDE_DataElementAdded))
            {
                new RoadVehicleTreeItem(this, child, NULL);
            }
        }
    }
}
