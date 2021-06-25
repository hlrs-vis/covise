/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   10/6/2010
**
**************************************************************************/

#include "roadsystemtreeitem.hpp"

// Data //
//
#include "src/data/roadsystem/roadsystem.hpp"
#include "src/data/roadsystem/rsystemelementroad.hpp"
#include "src/data/roadsystem/rsystemelementcontroller.hpp"
#include "src/data/roadsystem/rsystemelementjunction.hpp"
#include "src/data/roadsystem/rsystemelementfiddleyard.hpp"

// Tree //
//
#include "roadtreeitem.hpp"
#include "junctiontreeitem.hpp"
#include "controllertreeitem.hpp"

RoadSystemTreeItem::RoadSystemTreeItem(ProjectTree *projectTree, RoadSystem *roadSystem, QTreeWidgetItem *rootItem)
    : ProjectTreeItem(NULL, roadSystem, rootItem)
    , // no direct parent, rootItem is foster parent
    projectTree_(projectTree)
    , roadSystem_(roadSystem)
    , rootItem_(rootItem)
{
    init();
}

RoadSystemTreeItem::~RoadSystemTreeItem()
{
}

void
RoadSystemTreeItem::init()
{
    // Text //
    //
    setText(0, tr("RoadSystem"));

    // Roads //
    //
    roadsItem_ = new QTreeWidgetItem(this);
    roadsItem_->setText(0, tr("Roads"));

    foreach (RSystemElementRoad *road, getRoadSystem()->getRoads())
    {
        new RoadTreeItem(this, road, roadsItem_);
    }

    // Controllers //
    //
    controllersItem_ = new QTreeWidgetItem(this);
    controllersItem_->setText(0, tr("Controllers"));
    foreach (RSystemElementController *controller, getRoadSystem()->getControllers())
    {
        new ControllerTreeItem(this, controller, controllersItem_);
    }

    // Junctions //
    //
    junctionsItem_ = new QTreeWidgetItem(this);
    junctionsItem_->setText(0, tr("Junctions"));

    foreach (RSystemElementJunction *junction, getRoadSystem()->getJunctions())
    {
        new JunctionTreeItem(this, junction, junctionsItem_);
    }

    // Fiddleyards //
    //
    fiddleyardsItem_ = new QTreeWidgetItem(this);
    fiddleyardsItem_->setText(0, tr("Fiddleyards"));
    // TODO
}

//##################//
// Observer Pattern //
//##################//

void
RoadSystemTreeItem::updateObserver()
{
    // Parent //
    //
    ProjectTreeItem::updateObserver();

    if (isInGarbage())
    {
        return; // will be deleted anyway
    }

    // RoadSystem //
    //
    int changes = getRoadSystem()->getRoadSystemChanges();

    if (changes & RoadSystem::CRS_RoadChange)
    {
        // A road has been added (or deleted - but that will be handled by the road item itself).
        //
        foreach (RSystemElementRoad *road, getRoadSystem()->getRoads())
        {
            if ((road->getDataElementChanges() & DataElement::CDE_DataElementCreated)
                || (road->getDataElementChanges() & DataElement::CDE_DataElementAdded))
            {
                new RoadTreeItem(this, road, roadsItem_);
            }
        }
    }

    if (changes & RoadSystem::CRS_JunctionChange)
    {
        // A road has been added (or deleted - but that will be handled by the road item itself).
        //
        foreach (RSystemElementJunction *junction, getRoadSystem()->getJunctions())
        {
            if ((junction->getDataElementChanges() & DataElement::CDE_DataElementCreated)
                || (junction->getDataElementChanges() & DataElement::CDE_DataElementAdded))
            {
                new JunctionTreeItem(this, junction, junctionsItem_);
            }
        }
    }

    if (changes & RoadSystem::CRS_ControllerChange)
    {
        // A road has been added (or deleted - but that will be handled by the road item itself).
        //
        foreach (RSystemElementController *controller, getRoadSystem()->getControllers())
        {
            if ((controller->getDataElementChanges() & DataElement::CDE_DataElementCreated)
                || (controller->getDataElementChanges() & DataElement::CDE_DataElementAdded))
            {
                new ControllerTreeItem(this, controller, controllersItem_);
            }
        }
    }
}
