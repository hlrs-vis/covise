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

#include "pedestriangrouptreeitem.hpp"

// Data //
//
#include "src/data/pedestriansystem/pedestriangroup.hpp"
#include "src/data/pedestriansystem/pedestrian.hpp"

// Tree //
//
#include "pedestriansystemtreeitem.hpp"
#include "pedestriantreeitem.hpp"

PedestrianGroupTreeItem::PedestrianGroupTreeItem(PedestrianSystemTreeItem *parent, PedestrianGroup *group, QTreeWidgetItem *fosterParent)
    : ProjectTreeItem(parent, group, fosterParent)
    , pedestrianSystemTreeItem_(parent)
    , group_(group)
{
    init();
}

PedestrianGroupTreeItem::~PedestrianGroupTreeItem()
{
}

void
PedestrianGroupTreeItem::init()
{
    // Text //
    //
    updateText();

    // Pedestrians //
    //
    foreach (Pedestrian *child, group_->getPedestrians())
    {
        new PedestrianTreeItem(this, child, NULL);
    }
}

void
PedestrianGroupTreeItem::updateText()
{
    setText(0, tr("PedestrianGroup"));
}

//##################//
// Observer Pattern //
//##################//

void
PedestrianGroupTreeItem::updateObserver()
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
    int changes = group_->getPedestrianGroupChanges();
    if ((changes & PedestrianGroup::CVG_PedestrianChanged))
    {
        foreach (Pedestrian *child, group_->getPedestrians())
        {
            if ((child->getDataElementChanges() & DataElement::CDE_DataElementCreated)
                || (child->getDataElementChanges() & DataElement::CDE_DataElementAdded))
            {
                new PedestrianTreeItem(this, child, NULL);
            }
        }
    }
}
