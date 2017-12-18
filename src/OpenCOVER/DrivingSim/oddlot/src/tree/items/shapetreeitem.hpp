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

#ifndef SHAPETREEITEM_HPP
#define SHAPETREEITEM_HPP

#include "sectiontreeitem.hpp"

class RoadTreeItem;
class ShapeSection;

class ShapeSectionTreeItem : public SectionTreeItem
{
    Q_OBJECT

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit ShapeSectionTreeItem(RoadTreeItem *parent, ShapeSection *section, QTreeWidgetItem *fosterParent);
    virtual ~ShapeSectionTreeItem();

    // Road //
    //
    ShapeSection *getShapeSection() const
    {
        return shapeSection_;
    }

    // Obsever Pattern //
    //
    virtual void updateObserver();

protected:
    virtual void updateName();

private:
    ShapeSectionTreeItem(); /* not allowed */
    ShapeSectionTreeItem(const ShapeSectionTreeItem &); /* not allowed */
    ShapeSectionTreeItem &operator=(const ShapeSectionTreeItem &); /* not allowed */

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
    ShapeSection *shapeSection_;
};

#endif // SHAPETREEITEM_HPP
