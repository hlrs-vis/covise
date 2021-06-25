/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   22.04.2010
**
**************************************************************************/

#include "trackcompositeitem.hpp"

// Data //
//
#include "src/data/roadsystem/track/trackcomposite.hpp"
#include "src/data/commands/trackcommands.hpp"

// Graph //
//
//#include "trackroaditem.hpp"

// Editor //
//
//#include "src/graph/editors/trackeditor.hpp"

//################//
// CONSTRUCTOR    //
//################//

TrackCompositeItem::TrackCompositeItem(TrackRoadItem *parentTrackRoadItem, TrackComposite *trackComposite)
    : TrackComponentItem(parentTrackRoadItem, trackComposite)
    , trackComposite_(trackComposite)
{
    // Init //
    //
    init();
}

TrackCompositeItem::TrackCompositeItem(TrackComponentItem *parentTrackComponentItem, TrackComposite *trackComposite)
    : TrackComponentItem(parentTrackComponentItem, trackComposite)
    , trackComposite_(trackComposite)
{
    // Init //
    //
    init();
}

TrackCompositeItem::~TrackCompositeItem()
{
}

void
TrackCompositeItem::init()
{
    // Selection/Highlighting //
    //
    //setAcceptHoverEvents(true);
    //setFlag(QGraphicsItem::ItemIsMovable, true);
    setOpacitySettings(1.0, 1.0); // always highlighted
}

//################//
// SLOTS          //
//################//

void
TrackCompositeItem::ungroupComposite()
{
    UngroupTrackCompositeCommand *command = new UngroupTrackCompositeCommand(trackComposite_);
    getProjectGraph()->executeCommand(command);
}

//################//
// OBSERVER       //
//################//

void
TrackCompositeItem::updateObserver()
{
    // Parent //
    //
    TrackComponentItem::updateObserver();
    if (isInGarbage())
    {
        return; // will be deleted anyway
    }
}
