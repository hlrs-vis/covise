/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   10/19/2010
**
**************************************************************************/

#include "heightmapitem.hpp"

// Data //
//
#include "src/data/scenerysystem/heightmap.hpp"
#include "src/data/commands/scenerycommands.hpp"

// Graph //
//
#include "src/graph/items/scenerysystem/scenerysystemitem.hpp"

HeightmapItem::HeightmapItem(ScenerySystemItem *parentScenerySystem, Heightmap *heightmap)
    : SceneryMapItem(parentScenerySystem, heightmap)
    , heightmap_(heightmap)
{
    // Selection //
    //
    setSelectable();
}

HeightmapItem::~HeightmapItem()
{
}

//################//
// SLOTS          //
//################//

bool
HeightmapItem::removeMap()
{
    RemoveHeightmapCommand *command = new RemoveHeightmapCommand(heightmap_);
    return getProjectGraph()->executeCommand(command);
}

//################//
// EVENTS         //
//################//

/*!
* Handles Item Changes.
*/
//QVariant
//	HeightmapItem
//	::itemChange(GraphicsItemChange change, const QVariant & value)
//{

//}

//##################//
// Observer Pattern //
//##################//

void
HeightmapItem::updateObserver()
{
    // Parent //
    //
    SceneryMapItem::updateObserver();
    if (isInGarbage())
    {
        return; // will be deleted anyway
    }

    // Get change flags //
    //
    //	int changes = heightmap_->getHeightmapChanges();

    // Heightmap //
    //
    //	if((changes & Heightmap::C )
    //	{
    //	}
}
