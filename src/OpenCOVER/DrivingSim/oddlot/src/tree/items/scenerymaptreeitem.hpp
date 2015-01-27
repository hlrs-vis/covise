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

#ifndef SCENERYMAPTREEITEM_HPP
#define SCENERYMAPTREEITEM_HPP

#include "projecttreeitem.hpp"

class ScenerySystemTreeItem;
class SceneryMap;

class SceneryMapTreeItem : public ProjectTreeItem
{
    Q_OBJECT

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit SceneryMapTreeItem(ScenerySystemTreeItem *parent, SceneryMap *map, QTreeWidgetItem *fosterParent);
    virtual ~SceneryMapTreeItem();

    // SceneryMap //
    //
    SceneryMap *getSceneryMap() const
    {
        return map_;
    }

    // Obsever Pattern //
    //
    virtual void updateObserver();

protected:
private:
    SceneryMapTreeItem(); /* not allowed */
    SceneryMapTreeItem(const SceneryMapTreeItem &); /* not allowed */
    SceneryMapTreeItem &operator=(const SceneryMapTreeItem &); /* not allowed */

    void init();
    void updateText();

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
    ScenerySystemTreeItem *scenerySystemTreeItem_;

    // SceneryMap //
    //
    SceneryMap *map_;
};

#endif // SCENERYMAPTREEITEM_HPP
