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

#ifndef SECTIONTREEITEM_HPP
#define SECTIONTREEITEM_HPP

#include "projecttreeitem.hpp"

class RoadTreeItem;
class RoadSection;

class SectionTreeItem : public ProjectTreeItem
{
    Q_OBJECT

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit SectionTreeItem(RoadTreeItem *parent, RoadSection *roadSection_, QTreeWidgetItem *fosterParent);
    virtual ~SectionTreeItem();

    // Road //
    //
    RoadSection *getRoadSection() const
    {
        return roadSection_;
    }

    // Obsever Pattern //
    //
    virtual void updateObserver();

protected:
    virtual void updateName() = 0;

private:
    SectionTreeItem(); /* not allowed */
    SectionTreeItem(const SectionTreeItem &); /* not allowed */
    SectionTreeItem &operator=(const SectionTreeItem &); /* not allowed */

    void init();

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
    RoadTreeItem *roadTreeItem_;

    // Road //
    //
    RoadSection *roadSection_;
};

#endif // SECTIONTREEITEM_HPP
