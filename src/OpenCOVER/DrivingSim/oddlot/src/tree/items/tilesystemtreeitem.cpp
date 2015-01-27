/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   10/6/2010
**
**************************************************************************/

#include "tilesystemtreeitem.hpp"

// Data //
//
#include "src/data/tilesystem/tile.hpp"
#include "src/data/projectdata.hpp"

// Tree //
//
#include "tiletreeitem.hpp"

TileSystemTreeItem::TileSystemTreeItem(ProjectTree *projectTree, TileSystem *tileSystem, QTreeWidgetItem *rootItem)
    : ProjectTreeItem(NULL, tileSystem, rootItem)
    , // no direct parent, rootItem is foster parent
    projectTree_(projectTree)
    , tileSystem_(tileSystem)
    , rootItem_(rootItem)
    , tilesItem_(NULL)
{
    init();
}

TileSystemTreeItem::~TileSystemTreeItem()
{
}

void
TileSystemTreeItem::init()
{
    // Text //
    //
    setText(0, tr("Tiles"));

    //	tilesItem_ = new QTreeWidgetItem(this);

    // Tiles //
    //

    foreach (Tile *tile, getTileSystem()->getTiles())
    {
        new TileTreeItem(this, tile, tilesItem_);
    }
}

//##################//
// Observer Pattern //
//##################//

void
TileSystemTreeItem::updateObserver()
{
    // Parent //
    //
    ProjectTreeItem::updateObserver();

    if (isInGarbage())
    {
        return; // will be deleted anyway
    }

    // TileSystem //
    //
    int changes = getTileSystem()->getTileSystemChanges();

    if (changes & TileSystem::CTS_TileChange)
    {
        // A tile has been added (or deleted - but that will be handled by the tile item itself).
        //
        foreach (Tile *tile, getTileSystem()->getTiles())
        {
            if ((tile->getDataElementChanges() & DataElement::CDE_DataElementCreated)
                || (tile->getDataElementChanges() & DataElement::CDE_DataElementAdded))
            {
                new TileTreeItem(this, tile, tilesItem_);
            }
        }
    }
}
