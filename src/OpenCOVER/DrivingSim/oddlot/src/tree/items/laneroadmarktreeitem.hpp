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

#ifndef LANEROADMARKTREEITEM_HPP
#define LANEROADMARKTREEITEM_HPP

#include "projecttreeitem.hpp"

class LaneTreeItem;
class LaneRoadMark;

class LaneRoadMarkTreeItem : public ProjectTreeItem
{
    Q_OBJECT

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit LaneRoadMarkTreeItem(LaneTreeItem *parent, LaneRoadMark *laneRoadMark, QTreeWidgetItem *fosterParent);
    virtual ~LaneRoadMarkTreeItem();

    // Road //
    //
    LaneRoadMark *getLaneRoadMark() const
    {
        return laneRoadMark_;
    }

    // Obsever Pattern //
    //
    virtual void updateObserver();

private:
    LaneRoadMarkTreeItem(); /* not allowed */
    LaneRoadMarkTreeItem(const LaneRoadMarkTreeItem &); /* not allowed */
    LaneRoadMarkTreeItem &operator=(const LaneRoadMarkTreeItem &); /* not allowed */

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
    LaneRoadMark *laneRoadMark_;
};

#endif // LANEROADMARKTREEITEM_HPP
