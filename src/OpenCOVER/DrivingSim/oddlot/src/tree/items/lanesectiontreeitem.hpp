/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   10/13/2010
**
**************************************************************************/

#ifndef LANESECTIONTREEITEM_HPP
#define LANESECTIONTREEITEM_HPP

#include "sectiontreeitem.hpp"

class RoadTreeItem;
class LaneSection;

class LaneSectionTreeItem : public SectionTreeItem
{
    Q_OBJECT

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit LaneSectionTreeItem(RoadTreeItem *parent, LaneSection *section, QTreeWidgetItem *fosterParent);
    virtual ~LaneSectionTreeItem();

    // Road //
    //
    LaneSection *getLaneSection() const
    {
        return laneSection_;
    }

    // Obsever Pattern //
    //
    virtual void updateObserver();

protected:
    virtual void updateName();

private:
    LaneSectionTreeItem(); /* not allowed */
    LaneSectionTreeItem(const LaneSectionTreeItem &); /* not allowed */
    LaneSectionTreeItem &operator=(const LaneSectionTreeItem &); /* not allowed */

    void init();

    //################//
    // EVENTS         //
    //################//

public:
    //################//
    // PROPERTIES     //
    //################//

private:
    // Road //
    //
    LaneSection *laneSection_;
};

#endif // LANESECTIONTREEITEM_HPP
