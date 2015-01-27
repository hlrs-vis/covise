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

#ifndef TYPESECTIONTREEITEM_HPP
#define TYPESECTIONTREEITEM_HPP

#include "sectiontreeitem.hpp"

class RoadTreeItem;
class TypeSection;

class TypeSectionTreeItem : public SectionTreeItem
{
    Q_OBJECT

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit TypeSectionTreeItem(RoadTreeItem *parent, TypeSection *section, QTreeWidgetItem *fosterParent);
    virtual ~TypeSectionTreeItem();

    // Road //
    //
    TypeSection *getTypeSection() const
    {
        return typeSection_;
    }

    // Obsever Pattern //
    //
    virtual void updateObserver();

protected:
    virtual void updateName();

private:
    TypeSectionTreeItem(); /* not allowed */
    TypeSectionTreeItem(const TypeSectionTreeItem &); /* not allowed */
    TypeSectionTreeItem &operator=(const TypeSectionTreeItem &); /* not allowed */

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
    TypeSection *typeSection_;
};

#endif // TYPESECTIONTREEITEM_HPP
