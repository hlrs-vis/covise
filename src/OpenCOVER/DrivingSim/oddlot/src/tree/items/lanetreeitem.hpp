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

#ifndef LANETREEITEM_HPP
#define LANETREEITEM_HPP

#include "projecttreeitem.hpp"

class LaneSectionTreeItem;
class Lane;

class LaneTreeItem : public ProjectTreeItem
{
    Q_OBJECT

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit LaneTreeItem(LaneSectionTreeItem *parent, Lane *lane_, QTreeWidgetItem *fosterParent);
    virtual ~LaneTreeItem();

    // Road //
    //
    Lane *getLane() const
    {
        return lane_;
    }

    // Obsever Pattern //
    //
    virtual void updateObserver();

private:
    LaneTreeItem(); /* not allowed */
    LaneTreeItem(const LaneTreeItem &); /* not allowed */
    LaneTreeItem &operator=(const LaneTreeItem &); /* not allowed */

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
    LaneSectionTreeItem *laneSectionTreeItem_;

    // Road //
    //
    Lane *lane_;

    // Items //
    //
    QTreeWidgetItem *widthsItem_;
    QTreeWidgetItem *roadMarksItem_;
    QTreeWidgetItem *speedsItem_;
};

#endif // LANETREEITEM_HPP
