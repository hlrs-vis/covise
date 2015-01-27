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

#ifndef ROADSYSTEMTREEITEM_HPP
#define ROADSYSTEMTREEITEM_HPP

#include "projecttreeitem.hpp"

class RoadSystem;

class RoadSystemTreeItem : public ProjectTreeItem
{
    Q_OBJECT

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit RoadSystemTreeItem(ProjectTree *projectTree, RoadSystem *roadSystem, QTreeWidgetItem *rootItem);
    virtual ~RoadSystemTreeItem();

    // Tree //
    //
    virtual ProjectTree *getProjectTree() const
    {
        return projectTree_;
    }

    // RoadSystem //
    //
    RoadSystem *getRoadSystem() const
    {
        return roadSystem_;
    }

    // Obsever Pattern //
    //
    virtual void updateObserver();

protected:
private:
    RoadSystemTreeItem(); /* not allowed */
    RoadSystemTreeItem(const RoadSystemTreeItem &); /* not allowed */
    RoadSystemTreeItem &operator=(const RoadSystemTreeItem &); /* not allowed */

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
    RoadSystem *roadSystem_;

    // Items //
    //
    QTreeWidgetItem *rootItem_;

    QTreeWidgetItem *roadsItem_;
    QTreeWidgetItem *controllersItem_;
    QTreeWidgetItem *junctionsItem_;
    QTreeWidgetItem *fiddleyardsItem_;
};

#endif // ROADSYSTEMTREEITEM_HPP
