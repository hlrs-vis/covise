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

#include "roadvehicletreeitem.hpp"

// Data //
//
#include "src/data/vehiclesystem/roadvehicle.hpp"

// Tree //
//
#include "vehiclegrouptreeitem.hpp"

RoadVehicleTreeItem::RoadVehicleTreeItem(VehicleGroupTreeItem *parent, RoadVehicle *roadVehicle, QTreeWidgetItem *fosterParent)
    : ProjectTreeItem(parent, roadVehicle, fosterParent)
    , vehicleGroupTreeItem_(parent)
    , roadVehicle_(roadVehicle)
{
    init();
}

RoadVehicleTreeItem::~RoadVehicleTreeItem()
{
}

void
RoadVehicleTreeItem::init()
{
    updateText();
}

void
RoadVehicleTreeItem::updateText()
{
    // Text //
    //
    QString text = roadVehicle_->getId();
    if (!roadVehicle_->getName().isEmpty())
    {
        text.append(" (");
        text.append(roadVehicle_->getName());
        text.append(")");
    }
    setText(0, text);
}

//##################//
// Observer Pattern //
//##################//

void
RoadVehicleTreeItem::updateObserver()
{
    // Parent //
    //
    ProjectTreeItem::updateObserver();
    if (isInGarbage())
    {
        return; // will be deleted anyway
    }

    // TODO: RoadVehicle SUBJECT stuff

    // Get change flags //
    //
    //	int changes = roadVehicle_->get();
    //	if((changes & RoadVehicle::CSM_Id)
    //		|| (changes & RoadVehicle::CSM_Filename)
    //	)
    //	{
    //		updateText();
    //	}
}
