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

#ifndef SUPERELEVATIONTREEITEM_HPP
#define SUPERELEVATIONTREEITEM_HPP

#include "sectiontreeitem.hpp"

class RoadTreeItem;
class SuperelevationSection;

class SuperelevationSectionTreeItem : public SectionTreeItem
{
    Q_OBJECT

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit SuperelevationSectionTreeItem(RoadTreeItem *parent, SuperelevationSection *section, QTreeWidgetItem *fosterParent);
    virtual ~SuperelevationSectionTreeItem();

    // Road //
    //
    SuperelevationSection *getSuperelevationSection() const
    {
        return superelevationSection_;
    }

    // Obsever Pattern //
    //
    virtual void updateObserver();

protected:
    virtual void updateName();

private:
    SuperelevationSectionTreeItem(); /* not allowed */
    SuperelevationSectionTreeItem(const SuperelevationSectionTreeItem &); /* not allowed */
    SuperelevationSectionTreeItem &operator=(const SuperelevationSectionTreeItem &); /* not allowed */

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
    SuperelevationSection *superelevationSection_;
};

#endif // SUPERELEVATIONTREEITEM_HPP
