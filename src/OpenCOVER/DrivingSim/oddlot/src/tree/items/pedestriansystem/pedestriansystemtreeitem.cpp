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

#include "pedestriansystemtreeitem.hpp"

// Data //
//
#include "src/data/pedestriansystem/pedestriansystem.hpp"
#include "src/data/pedestriansystem/pedestriangroup.hpp"

// Tree //
//
#include "pedestriangrouptreeitem.hpp"

PedestrianSystemTreeItem::PedestrianSystemTreeItem(ProjectTree *projectTree, PedestrianSystem *pedestrianSystem, QTreeWidgetItem *rootItem)
    : ProjectTreeItem(NULL, pedestrianSystem, rootItem)
    , // no direct parent, rootItem is foster parent
    projectTree_(projectTree)
    , pedestrianSystem_(pedestrianSystem)
    , rootItem_(rootItem)
{
    init();
}

PedestrianSystemTreeItem::~PedestrianSystemTreeItem()
{
}

void
PedestrianSystemTreeItem::init()
{
    // Text //
    //
    setText(0, tr("PedestrianSystem"));

    // Maps //
    //
    pedestrianGroupsItem_ = new QTreeWidgetItem(this);
    pedestrianGroupsItem_->setText(0, tr("Pedestrian Groups"));

    foreach (PedestrianGroup *group, pedestrianSystem_->getPedestrianGroups())
    {
        new PedestrianGroupTreeItem(this, group, pedestrianGroupsItem_);
    }
}

//##################//
// Observer Pattern //
//##################//

void
PedestrianSystemTreeItem::updateObserver()
{
    // Parent //
    //
    ProjectTreeItem::updateObserver();

    if (isInGarbage())
    {
        return; // will be deleted anyway
    }

    // PedestrianSystem //
    //
    int changes = getPedestrianSystem()->getPedestrianSystemChanges();

    if (changes & PedestrianSystem::CVS_PedestrianGroupsChanged)
    {
        // A PedestrianGroup has been added (or deleted - but that will be handled by the item itself).
        //
        foreach (PedestrianGroup *group, pedestrianSystem_->getPedestrianGroups())
        {
            if ((group->getDataElementChanges() & DataElement::CDE_DataElementCreated)
                || (group->getDataElementChanges() & DataElement::CDE_DataElementAdded))
            {
                new PedestrianGroupTreeItem(this, group, pedestrianGroupsItem_);
            }
        }
    }
}
