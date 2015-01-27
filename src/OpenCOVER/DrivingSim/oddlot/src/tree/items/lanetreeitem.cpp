/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   10/29/2010
**
**************************************************************************/

#include "lanetreeitem.hpp"

// Data //
//
#include "src/data/roadsystem/sections/lane.hpp"
#include "src/data/roadsystem/sections/lanewidth.hpp"
#include "src/data/roadsystem/sections/lanespeed.hpp"
#include "src/data/roadsystem/sections/laneroadmark.hpp"

// Tree //
//
#include "lanesectiontreeitem.hpp"
#include "lanewidthtreeitem.hpp"
#include "laneroadmarktreeitem.hpp"
#include "lanespeedtreeitem.hpp"

LaneTreeItem::LaneTreeItem(LaneSectionTreeItem *parent, Lane *lane, QTreeWidgetItem *fosterParent)
    : ProjectTreeItem(parent, lane, fosterParent)
    , laneSectionTreeItem_(parent)
    , lane_(lane)
{
    init();
}

LaneTreeItem::~LaneTreeItem()
{
}

void
LaneTreeItem::init()
{
    updateName();

    // LaneWidths //
    //
    widthsItem_ = new QTreeWidgetItem(this);
    widthsItem_->setText(0, tr("widths"));
    foreach (LaneWidth *element, lane_->getWidthEntries())
    {
        new LaneWidthTreeItem(this, element, widthsItem_);
    }

    // LaneRoadMarks //
    //
    roadMarksItem_ = new QTreeWidgetItem(this);
    roadMarksItem_->setText(0, tr("roadMarks"));
    foreach (LaneRoadMark *element, lane_->getRoadMarkEntries())
    {
        new LaneRoadMarkTreeItem(this, element, roadMarksItem_);
    }

    // LaneSpeeds //
    //
    speedsItem_ = new QTreeWidgetItem(this);
    speedsItem_->setText(0, tr("speeds"));
    foreach (LaneSpeed *element, lane_->getSpeedEntries())
    {
        new LaneSpeedTreeItem(this, element, speedsItem_);
    }
}

void
LaneTreeItem::updateName()
{
    QString text(tr("lane "));
    if (lane_->getId() < 0)
    {
        text.append(QString("%1: ").arg(lane_->getId()));
    }
    else
    {
        text.append(QString(" %1: ").arg(lane_->getId()));
    }
    text.append(Lane::parseLaneTypeBack(lane_->getLaneType()));
    setText(0, text);
}

//##################//
// Observer Pattern //
//##################//

void
LaneTreeItem::updateObserver()
{

    // Parent //
    //
    ProjectTreeItem::updateObserver();

    if (isInGarbage())
    {
        return; // will be deleted anyway
    }

    // Lane //
    //
    int changes = lane_->getLaneChanges();
    if ((changes & Lane::CLN_IdChanged)
        || (changes & Lane::CLN_TypeChanged))
    {
        // Change of the road coordinate s //
        //
        updateName();
    }

    // LaneWidths //
    //
    if (changes & Lane::CLN_WidthsChanged)
    {
        foreach (LaneWidth *element, lane_->getWidthEntries())
        {
            if ((element->getDataElementChanges() & DataElement::CDE_DataElementCreated)
                || (element->getDataElementChanges() & DataElement::CDE_DataElementAdded))
            {
                new LaneWidthTreeItem(this, element, widthsItem_);
            }
        }
    }

    // LaneRoadMarks //
    //
    if (changes & Lane::CLN_RoadMarksChanged)
    {
        foreach (LaneRoadMark *element, lane_->getRoadMarkEntries())
        {
            if ((element->getDataElementChanges() & DataElement::CDE_DataElementCreated)
                || (element->getDataElementChanges() & DataElement::CDE_DataElementAdded))
            {
                new LaneRoadMarkTreeItem(this, element, roadMarksItem_);
            }
        }
    }

    // LaneSpeeds //
    //
    if (changes & Lane::CLN_SpeedsChanged)
    {
        foreach (LaneSpeed *element, lane_->getSpeedEntries())
        {
            if ((element->getDataElementChanges() & DataElement::CDE_DataElementCreated)
                || (element->getDataElementChanges() & DataElement::CDE_DataElementAdded))
            {
                new LaneSpeedTreeItem(this, element, speedsItem_);
            }
        }
    }
}
