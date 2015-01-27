/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   10/26/2010
**
**************************************************************************/

#include "trackcomponenttreeitem.hpp"

// Data //
//
#include "src/data/roadsystem/track/trackcomponent.hpp"

// Tree //
//
#include "roadtreeitem.hpp"

TrackComponentTreeItem::TrackComponentTreeItem(RoadTreeItem *parent, TrackComponent *component, QTreeWidgetItem *fosterParent)
    : ProjectTreeItem(parent, component, fosterParent)
    , parentRoadTreeItem_(parent)
    , parentTrackComponentTreeItem_(NULL)
    , trackComponent_(component)
{
    init();
}

TrackComponentTreeItem::TrackComponentTreeItem(TrackComponentTreeItem *parent, TrackComponent *component, QTreeWidgetItem *fosterParent)
    : ProjectTreeItem(parent, component, fosterParent)
    , parentRoadTreeItem_(NULL)
    , parentTrackComponentTreeItem_(parent)
    , trackComponent_(component)
{
    init();
}

TrackComponentTreeItem::~TrackComponentTreeItem()
{
}

void
TrackComponentTreeItem::init()
{
    // Text //
    //
    TrackComponent::DTrackType type = trackComponent_->getTrackType();
    if (type == TrackComponent::DTT_LINE)
    {
        setText(0, tr("line"));
    }
    else if (type == TrackComponent::DTT_ARC)
    {
        setText(0, tr("arc"));
    }
    else if (type == TrackComponent::DTT_SPIRAL)
    {
        setText(0, tr("spiral"));
    }
    else if (type == TrackComponent::DTT_POLY3)
    {
        setText(0, tr("poly3"));
    }
    else if (type == TrackComponent::DTT_COMPOSITE)
    {
        setText(0, tr("composite"));
    }
    else if (type == TrackComponent::DTT_SPARCS)
    {
        setText(0, tr("spiral-arc-spiral"));
    }
    else
    {
        setText(0, tr("unknown type"));
    }

    // Children //
    //
    foreach (TrackComponent *element, trackComponent_->getChildTrackComponents())
    {
        new TrackComponentTreeItem(this, element, NULL);
    }
}

//##################//
// Observer Pattern //
//##################//

void
TrackComponentTreeItem::updateObserver()
{

    // Parent //
    //
    ProjectTreeItem::updateObserver();

    if (isInGarbage())
    {
        return; // will be deleted anyway
    }

    // RoadSection //
    //
    int changes = trackComponent_->getTrackComponentChanges();
    //	if(changes & TrackComponent::CT)
    //	{
    //		foreach(TrackComponent * element, trackComponent_->getChildTrackComponents())
    //		{
    //			if((element->getDataElementChanges() & DataElement::CDE_DataElementCreated)
    //				|| (element->getDataElementChanges() & DataElement::CDE_DataElementAdded)
    //				)
    //			{
    //				new TrackComponentTreeItem(this, element, NULL);
    //			}
    //		}
    //	}
}
