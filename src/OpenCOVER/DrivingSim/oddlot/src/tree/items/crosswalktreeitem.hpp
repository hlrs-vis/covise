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

#ifndef CROSSWALKTREEITEM_HPP
#define CROSSWALKTREEITEM_HPP

#include "sectiontreeitem.hpp"

class RoadTreeItem;
class Crosswalk;

class CrosswalkTreeItem : public SectionTreeItem
{
    Q_OBJECT

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit CrosswalkTreeItem(RoadTreeItem *parent, Crosswalk *section, QTreeWidgetItem *fosterParent);
    virtual ~CrosswalkTreeItem();

    // CrosswalkCrosswalk //
    //
    Crosswalk *getCrosswalk() const
    {
        return crosswalk_;
    }

    // Obsever Pattern //
    //
    virtual void updateObserver();

protected:
    virtual void updateName();

private:
    CrosswalkTreeItem(); /* not allowed */
    CrosswalkTreeItem(const CrosswalkTreeItem &); /* not allowed */
    CrosswalkTreeItem &operator=(const CrosswalkTreeItem &); /* not allowed */

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
    Crosswalk *crosswalk_;
};

#endif // CROSSWALKTREEITEM_HPP
