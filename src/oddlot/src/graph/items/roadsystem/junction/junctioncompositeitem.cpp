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

#include "junctioncompositeitem.hpp"

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

JunctionCompositeItem::JunctionCompositeItem(JunctionRoadItem *parentJunctionRoadItem, TrackComposite *trackComposite)
    : JunctionComponentItem(parentJunctionRoadItem, trackComposite)
    , trackComposite_(trackComposite)
{
    // Init //
    //
    init();
}

JunctionCompositeItem::JunctionCompositeItem(JunctionComponentItem *parentJunctionComponentItem, TrackComposite *trackComposite)
    : JunctionComponentItem(parentJunctionComponentItem, trackComposite)
    , trackComposite_(trackComposite)
{
    // Init //
    //
    init();
}

JunctionCompositeItem::~JunctionCompositeItem()
{
}

void
JunctionCompositeItem::init()
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
JunctionCompositeItem::ungroupComposite()
{
    UngroupTrackCompositeCommand *command = new UngroupTrackCompositeCommand(trackComposite_);
    getProjectGraph()->executeCommand(command);
}

//################//
// OBSERVER       //
//################//

void
JunctionCompositeItem::updateObserver()
{
    // Parent //
    //
    JunctionComponentItem::updateObserver();
    if (isInGarbage())
    {
        return; // will be deleted anyway
    }
}
