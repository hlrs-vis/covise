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

#include "elevationsectiontreeitem.hpp"

// Data //
//
#include "src/data/roadsystem/sections/elevationsection.hpp"

// Tree //
//
#include "roadtreeitem.hpp"

ElevationSectionTreeItem::ElevationSectionTreeItem(RoadTreeItem *parent, ElevationSection *section, QTreeWidgetItem *fosterParent)
    : SectionTreeItem(parent, section, fosterParent)
    , elevationSection_(section)
{
    init();
}

ElevationSectionTreeItem::~ElevationSectionTreeItem()
{
}

void
ElevationSectionTreeItem::init()
{
    updateName();
}

void
ElevationSectionTreeItem::updateName()
{
    QString text(tr("ElevationSection (%1)").arg(elevationSection_->getDegree()));
    text.append(QString(" %1").arg(elevationSection_->getSStart()));
    setText(0, text);
}

//##################//
// Observer Pattern //
//##################//

void
ElevationSectionTreeItem::updateObserver()
{

    // Parent //
    //
    SectionTreeItem::updateObserver();

    if (isInGarbage())
    {
        return; // will be deleted anyway
    }

    // ElevationSection //
    //
    int changes = elevationSection_->getElevationSectionChanges();

    if (changes & ElevationSection::CEL_ParameterChange)
    {
        updateName();
    }
}
