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

#ifndef OBJECTTREEITEM_HPP
#define OBJECTTREEITEM_HPP

#include "sectiontreeitem.hpp"

class RoadTreeItem;
class Object;

class ObjectTreeItem : public SectionTreeItem
{
    Q_OBJECT

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit ObjectTreeItem(RoadTreeItem *parent, Object *section, QTreeWidgetItem *fosterParent);
    virtual ~ObjectTreeItem();

    // ObjectObject //
    //
    Object *getObject() const
    {
        return object_;
    }

    // Obsever Pattern //
    //
    virtual void updateObserver();

protected:
    virtual void updateName();

private:
    ObjectTreeItem(); /* not allowed */
    ObjectTreeItem(const ObjectTreeItem &); /* not allowed */
    ObjectTreeItem &operator=(const ObjectTreeItem &); /* not allowed */

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
    Object *object_;
};

#endif // OBJECTTREEITEM_HPP
