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

#include "tiletreeitem.hpp"

// Data //
//
#include "src/data/tilesystem/tile.hpp"
#include "src/data/projectdata.hpp"

// Tree //
//
#include "tilesystemtreeitem.hpp"

TileTreeItem::TileTreeItem(TileSystemTreeItem *parent, Tile *tile, QTreeWidgetItem *fosterParent)
    : ProjectTreeItem(parent, tile, fosterParent)
    , tileSystemTreeItem_(parent)
    , tile_(tile)
{
    init();
}

TileTreeItem::~TileTreeItem()
{
}

void
TileTreeItem::init()
{
    // Text //
    //
    updateName();
}

void
TileTreeItem::updateName()
{
    QString text = tile_->getName();
    text.append(" (");
    text.append(tile_->getID());
    text.append(")");

    setText(0, text);
}

//##################//
// Observer Pattern //
//##################//

void
TileTreeItem::updateObserver()
{

    // Parent //
    //
    ProjectTreeItem::updateObserver();
    if (isInGarbage())
    {
        return; // will be deleted anyway
    }

    // Tile //
    //
    int changes = tile_->getTileChanges();

    if ((changes & Tile::CT_NameChange)
        || (changes & Tile::CT_IdChange))
    {
        updateName();
    }

    // Data //
    //
    int dataElementChanges = tile_->getDataElementChanges();

    if ((dataElementChanges & DataElement::CDE_SelectionChange)
        || (dataElementChanges & DataElement::CDE_ChildSelectionChange))
    {
        if (tile_->isElementSelected())
        {
            tile_->getTileSystem()->setCurrentTile(tile_);
        }
    }
}
