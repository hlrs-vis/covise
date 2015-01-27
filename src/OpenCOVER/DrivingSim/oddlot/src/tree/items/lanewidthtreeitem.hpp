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

#ifndef LANEWIDTHTREEITEM_HPP
#define LANEWIDTHTREEITEM_HPP

#include "projecttreeitem.hpp"

class LaneTreeItem;
class LaneWidth;

class LaneWidthTreeItem : public ProjectTreeItem
{
    Q_OBJECT

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit LaneWidthTreeItem(LaneTreeItem *parent, LaneWidth *laneWidth_, QTreeWidgetItem *fosterParent);
    virtual ~LaneWidthTreeItem();

    // Road //
    //
    LaneWidth *getLaneWidth() const
    {
        return laneWidth_;
    }

    // Obsever Pattern //
    //
    virtual void updateObserver();

private:
    LaneWidthTreeItem(); /* not allowed */
    LaneWidthTreeItem(const LaneWidthTreeItem &); /* not allowed */
    LaneWidthTreeItem &operator=(const LaneWidthTreeItem &); /* not allowed */

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
    LaneWidth *laneWidth_;
};

#endif // LANEWIDTHTREEITEM_HPP
