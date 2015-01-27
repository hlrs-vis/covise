/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   10/13/2010
**
**************************************************************************/

#include "crosswalktreeitem.hpp"

// Data //
//
#include "src/data/roadsystem/sections/crosswalkobject.hpp"

// Tree //
//
#include "roadtreeitem.hpp"

CrosswalkTreeItem::CrosswalkTreeItem(RoadTreeItem *parent, Crosswalk *crosswalk, QTreeWidgetItem *fosterParent)
    : SectionTreeItem(parent, crosswalk, fosterParent)
    , crosswalk_(crosswalk)
{
    init();
}

CrosswalkTreeItem::~CrosswalkTreeItem()
{
}

void
CrosswalkTreeItem::init()
{
    updateName();
}

void
CrosswalkTreeItem::updateName()
{
    QString text = crosswalk_->getName();
    text.append(" (");
    text.append(crosswalk_->getType());
    text.append(QString(") %1").arg(crosswalk_->getS()));
    setText(0, text);
}

//##################//
// Observer Pattern //
//##################//

void
CrosswalkTreeItem::updateObserver()
{

    // Parent //
    //
    SectionTreeItem::updateObserver();

    if (isInGarbage())
    {
        return; // will be deleted anyway
    }

    // Crosswalk //
    //
    int changes = crosswalk_->getCrosswalkChanges();

    if (changes & Crosswalk::CEL_ParameterChange)
    {
        updateName();
    }
}
