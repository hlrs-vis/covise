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

#include "superelevationtreeitem.hpp"

// Data //
//
#include "src/data/roadsystem/sections/superelevationsection.hpp"

// Tree //
//
#include "roadtreeitem.hpp"

SuperelevationSectionTreeItem::SuperelevationSectionTreeItem(RoadTreeItem *parent, SuperelevationSection *section, QTreeWidgetItem *fosterParent)
    : SectionTreeItem(parent, section, fosterParent)
    , superelevationSection_(section)
{
    init();
}

SuperelevationSectionTreeItem::~SuperelevationSectionTreeItem()
{
}

void
SuperelevationSectionTreeItem::init()
{
    updateName();
}

void
SuperelevationSectionTreeItem::updateName()
{
    QString text(tr("SuperelevationSection (%1)").arg(superelevationSection_->getDegree()));
    text.append(QString(" %1").arg(superelevationSection_->getSStart()));
    setText(0, text);
}

//##################//
// Observer Pattern //
//##################//

void
SuperelevationSectionTreeItem::updateObserver()
{

    // Parent //
    //
    SectionTreeItem::updateObserver();

    if (isInGarbage())
    {
        return; // will be deleted anyway
    }

    // SuperelevationSection //
    //
    int changes = superelevationSection_->getSuperelevationSectionChanges();

    if (changes & SuperelevationSection::CSE_ParameterChange)
    {
        updateName();
    }
}
