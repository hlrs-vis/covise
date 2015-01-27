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

#include "lanesectiontreeitem.hpp"

// Data //
//
#include "src/data/roadsystem/sections/lanesection.hpp"
#include "src/data/roadsystem/sections/lane.hpp"

// Tree //
//
#include "roadtreeitem.hpp"
#include "lanetreeitem.hpp"

LaneSectionTreeItem::LaneSectionTreeItem(RoadTreeItem *parent, LaneSection *section, QTreeWidgetItem *fosterParent)
    : SectionTreeItem(parent, section, fosterParent)
    , laneSection_(section)
{
    init();
}

LaneSectionTreeItem::~LaneSectionTreeItem()
{
}

void
LaneSectionTreeItem::init()
{
    updateName();

    foreach (Lane *element, laneSection_->getLanes())
    {
        new LaneTreeItem(this, element, NULL);
    }
}

void
LaneSectionTreeItem::updateName()
{
    QString text(tr("LaneSection"));
    text.append(QString(" %1").arg(laneSection_->getSStart()));
    setText(0, text);
}

//##################//
// Observer Pattern //
//##################//

void
LaneSectionTreeItem::updateObserver()
{

    // Parent //
    //
    SectionTreeItem::updateObserver(); // Handles change of the road coordinate s => updateName()

    if (isInGarbage())
    {
        return; // will be deleted anyway
    }

    // LaneSection //
    //
    int changes = laneSection_->getLaneSectionChanges();

    if (changes & LaneSection::CLS_LanesChanged)
    {
        // Lanes //
        //
        foreach (Lane *element, laneSection_->getLanes())
        {
            if ((element->getDataElementChanges() & DataElement::CDE_DataElementCreated)
                || (element->getDataElementChanges() & DataElement::CDE_DataElementAdded))
            {
                new LaneTreeItem(this, element, NULL);
            }
        }
    }
}
