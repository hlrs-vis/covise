/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   10/7/2010
**
**************************************************************************/

#ifndef SCENERYSYSTEMTREEITEM_HPP
#define SCENERYSYSTEMTREEITEM_HPP

#include "projecttreeitem.hpp"

class ScenerySystem;

class ScenerySystemTreeItem : public ProjectTreeItem
{
    Q_OBJECT

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit ScenerySystemTreeItem(ProjectTree *projectTree, ScenerySystem *scenerySystem, QTreeWidgetItem *rootItem);
    virtual ~ScenerySystemTreeItem();

    // Tree //
    //
    virtual ProjectTree *getProjectTree() const
    {
        return projectTree_;
    }

    // ScenerySystem //
    //
    ScenerySystem *getScenerySystem() const
    {
        return scenerySystem_;
    }

    // Obsever Pattern //
    //
    virtual void updateObserver();

protected:
private:
    ScenerySystemTreeItem(); /* not allowed */
    ScenerySystemTreeItem(const ScenerySystemTreeItem &); /* not allowed */
    ScenerySystemTreeItem &operator=(const ScenerySystemTreeItem &); /* not allowed */

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

    // ScenerySystem //
    //
    ScenerySystem *scenerySystem_;

    // Items //
    //
    QTreeWidgetItem *rootItem_;

    QTreeWidgetItem *aerialMapsItem_;
    QTreeWidgetItem *heightMapsItem_;
};

#endif // SCENERYSYSTEMTREEITEM_HPP
