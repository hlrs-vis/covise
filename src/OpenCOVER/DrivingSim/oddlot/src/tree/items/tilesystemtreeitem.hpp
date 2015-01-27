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

#ifndef TILESYSTEMTREEITEM_HPP
#define TILESYSTEMTREEITEM_HPP

#include "projecttreeitem.hpp"

class TileSystem;

class TileSystemTreeItem : public ProjectTreeItem
{
    Q_OBJECT

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit TileSystemTreeItem(ProjectTree *projectTree, TileSystem *tileSystem, QTreeWidgetItem *rootItem);
    virtual ~TileSystemTreeItem();

    // Tree //
    //
    virtual ProjectTree *getProjectTree() const
    {
        return projectTree_;
    }

    // RoadSystem //
    //
    TileSystem *getTileSystem() const
    {
        return tileSystem_;
    }

    // Obsever Pattern //
    //
    virtual void updateObserver();

protected:
private:
    TileSystemTreeItem(); /* not allowed */
    TileSystemTreeItem(const TileSystemTreeItem &); /* not allowed */
    TileSystemTreeItem &operator=(const TileSystemTreeItem &); /* not allowed */

    void init();

    //################//
    // EVENTS         //
    //################//

public:
    //################//
    // PROPERTIES     //
    //################//

private:
    // Tree //
    //
    ProjectTree *projectTree_;

    // RoadSystem //
    //
    TileSystem *tileSystem_;

    // Items //
    //
    QTreeWidgetItem *rootItem_;

    QTreeWidgetItem *tilesItem_;
};

#endif // TILESYSTEMTREEITEM_HPP
