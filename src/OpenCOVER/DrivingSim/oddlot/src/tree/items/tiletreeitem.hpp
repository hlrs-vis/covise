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

#ifndef TILETREEITEM_HPP
#define TILETREEITEM_HPP

#include "projecttreeitem.hpp"

class TileSystemTreeItem;
class Tile;

class TileTreeItem : public ProjectTreeItem
{
    Q_OBJECT

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit TileTreeItem(TileSystemTreeItem *parent, Tile *tile, QTreeWidgetItem *fosterParent);
    virtual ~TileTreeItem();

    // Tile //
    //
    Tile *getTile() const
    {
        return tile_;
    }

    // Obsever Pattern //
    //
    virtual void updateObserver();

protected:
private:
    TileTreeItem(); /* not allowed */
    TileTreeItem(const TileTreeItem &); /* not allowed */
    TileTreeItem &operator=(const TileTreeItem &); /* not allowed */

    void init();

    void updateName();

    //################//
    // EVENTS         //
    //################//

public:
    //################//
    // PROPERTIES     //
    //################//

private:
    // Parent //
    //
    TileSystemTreeItem *tileSystemTreeItem_;

    // Road //
    //
    Tile *tile_;
};

#endif // TILETREEITEM_HPP
