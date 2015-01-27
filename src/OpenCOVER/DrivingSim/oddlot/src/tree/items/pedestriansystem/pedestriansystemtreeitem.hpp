/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   11/17/2010
**
**************************************************************************/

#ifndef PEDESTRIANSYSTEMTREEITEM_HPP
#define PEDESTRIANSYSTEMTREEITEM_HPP

#include "src/tree/items/projecttreeitem.hpp"

class PedestrianSystem;

class PedestrianSystemTreeItem : public ProjectTreeItem
{
    Q_OBJECT

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit PedestrianSystemTreeItem(ProjectTree *projectTree, PedestrianSystem *pedestrianSystem, QTreeWidgetItem *rootItem);
    virtual ~PedestrianSystemTreeItem();

    // Tree //
    //
    virtual ProjectTree *getProjectTree() const
    {
        return projectTree_;
    }

    // PedestrianSystem //
    //
    PedestrianSystem *getPedestrianSystem() const
    {
        return pedestrianSystem_;
    }

    // Obsever Pattern //
    //
    virtual void updateObserver();

protected:
private:
    PedestrianSystemTreeItem(); /* not allowed */
    PedestrianSystemTreeItem(const PedestrianSystemTreeItem &); /* not allowed */
    PedestrianSystemTreeItem &operator=(const PedestrianSystemTreeItem &); /* not allowed */

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

    // PedestrianSystem //
    //
    PedestrianSystem *pedestrianSystem_;

    // Items //
    //
    QTreeWidgetItem *rootItem_;

    QTreeWidgetItem *pedestrianGroupsItem_;
};

#endif // PEDESTRIANSYSTEMTREEITEM_HPP
