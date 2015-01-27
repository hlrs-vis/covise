/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   10/29/2010
**
**************************************************************************/

#ifndef LANESPEEDTREEITEM_HPP
#define LANESPEEDTREEITEM_HPP

#include "projecttreeitem.hpp"

class LaneTreeItem;
class LaneSpeed;

class LaneSpeedTreeItem : public ProjectTreeItem
{
    Q_OBJECT

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit LaneSpeedTreeItem(LaneTreeItem *parent, LaneSpeed *laneSpeed_, QTreeWidgetItem *fosterParent);
    virtual ~LaneSpeedTreeItem();

    // Road //
    //
    LaneSpeed *getLaneSpeed() const
    {
        return laneSpeed_;
    }

    // Obsever Pattern //
    //
    virtual void updateObserver();

private:
    LaneSpeedTreeItem(); /* not allowed */
    LaneSpeedTreeItem(const LaneSpeedTreeItem &); /* not allowed */
    LaneSpeedTreeItem &operator=(const LaneSpeedTreeItem &); /* not allowed */

    void init();
    virtual void updateName();

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
    LaneTreeItem *laneTreeItem_;

    // Road //
    //
    LaneSpeed *laneSpeed_;
};

#endif // LANESPEEDTREEITEM_HPP
