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

#ifndef ELEVATIONSECTIONTREEITEM_HPP
#define ELEVATIONSECTIONTREEITEM_HPP

#include "sectiontreeitem.hpp"

class RoadTreeItem;
class ElevationSection;

class ElevationSectionTreeItem : public SectionTreeItem
{
    Q_OBJECT

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit ElevationSectionTreeItem(RoadTreeItem *parent, ElevationSection *section, QTreeWidgetItem *fosterParent);
    virtual ~ElevationSectionTreeItem();

    // Road //
    //
    ElevationSection *getElevationSection() const
    {
        return elevationSection_;
    }

    // Obsever Pattern //
    //
    virtual void updateObserver();

protected:
    virtual void updateName();

private:
    ElevationSectionTreeItem(); /* not allowed */
    ElevationSectionTreeItem(const ElevationSectionTreeItem &); /* not allowed */
    ElevationSectionTreeItem &operator=(const ElevationSectionTreeItem &); /* not allowed */

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
    ElevationSection *elevationSection_;
};

#endif // ELEVATIONSECTIONTREEITEM_HPP
